'''
YoloExt: YOLO inference in TouchDesigner via Ultralytics.
No auto-load: init only sets up component. Load engine by Pulse par "Loadengine" (parexec -> Loadengine()).
Inference runs only when par Inference = True.
- table1: detections (det_id, class, confidence, x, y, width, height) in pixels
- table2: pose keypoints when using pose model (det_id, kpt_id, x, y, visibility)
- scriptOp: segment masks (RGBA) when using segment model — combined mask in R,G,B, A=1
'''
import os
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
import webbrowser

try:
	import cv2
except ImportError:
	cv2 = None


def _parse_engine_dims(filename):
	"""Parse WxH from engine filename e.g. yolo11n_640x640_b1_fp16.engine -> (640, 640)."""
	m = re.search(r"(\d+)x(\d+)", filename)
	if m:
		return int(m.group(1)), int(m.group(2))
	return 640, 640


class YoloExt:
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.ownerComp.par.Dimensions = ''
		self.device = "cuda"
		self.model = None
		self.stream = None
		self.engine_h = 640
		self.engine_w = 640
		self.trt_path = ""
		try:
			self.conf_threshold = float(self.ownerComp.par.Confidence.val or 0.25)
		except Exception:
			self.conf_threshold = 0.25
		try:
			self.iou_threshold = float(self.ownerComp.par.Iou.val or 0.45)
		except Exception:
			self.iou_threshold = 0.45
		try:
			self.kpt_conf_threshold = float(self.ownerComp.par.Keypointconfidence.val or 0.25)
		except Exception:
			self.kpt_conf_threshold = 0.25

		try:
			self.source = op('null1')
			self.to_tensor = TopArrayInterface(self.source)
		except Exception:
			self.source = None
			self.to_tensor = None

	def Loadengine(self):
		"""Unload old pipeline, load new engine. Pulse par Loadengine triggers this."""
		if self.model is not None:
			del self.model
			self.model = None
		self.stream = None
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		try:
			self.source = op('null1')
			self.to_tensor = TopArrayInterface(self.source)
		except Exception:
			self.source = None
			self.to_tensor = None

		env_path = self.ownerComp.par.Venvpath.val or ""
		engine_file = self.ownerComp.par.Enginefile.val or ""
		if not env_path or not engine_file:
			self.trt_path = ""
			self.ownerComp.par.Dimensions = ""
			return
		self.trt_path = os.path.normpath(os.path.join(env_path, "engines", "yolo", engine_file))
		if not os.path.isfile(self.trt_path):
			debug(f"Engine file not found: {self.trt_path}")
			self.ownerComp.par.Dimensions = ""
			return
		try:
			try:
				task = str(self.ownerComp.par.Task.eval() or self.ownerComp.par.Task.val or 'detect').strip().lower()
			except Exception:
				task = 'detect'
			if task not in ('detect', 'segment', 'classify', 'pose', 'obb'):
				task = 'detect'
			repo = os.path.normpath(env_path)
			if repo not in sys.path:
				sys.path.insert(0, repo)
			from ultralytics import YOLO
			self.model = YOLO(self.trt_path, task=task)
			self.stream = torch.cuda.current_stream(device=self.device)
			self.engine_w, self.engine_h = _parse_engine_dims(engine_file)
			self.ownerComp.par.Dimensions = f"{self.engine_w}x{self.engine_h}"
			try:
				op('fit2').par.resolutionw = self.engine_w
				op('fit2').par.resolutionh = self.engine_h
			except Exception:
				pass
		except Exception as e:
			debug(e)
			self.model = None
			self.ownerComp.par.Dimensions = ""

	def parexec_onPulse(self, par):
		"""Called by Parameter Execute when a Pulse par fires. Loadengine = reload engine."""
		if par.name == "Loadengine":
			self.Loadengine()

	def parexec_onValueChange(self, par, prev):
		"""Called by Parameter Execute when a parameter value changes."""
		if par.name == "Venvpath" or par.name == "Enginefile":
			pass
		elif par.name == "Confidence":
			try:
				self.conf_threshold = float(par.val or 0.25)
			except Exception:
				pass
		elif par.name == "Iou":
			try:
				self.iou_threshold = float(par.val or 0.45)
			except Exception:
				pass
		elif par.name == "Keypointconfidence":
			try:
				self.kpt_conf_threshold = float(par.val or 0.25)
			except Exception:
				pass

	def preprocess_image(self, image):
		"""(C, H, W) -> (1, 3, engine_h, engine_w) with letterbox (proportional scale + pad)."""
		image = image[:3, :, :]
		src_h, src_w = image.shape[1], image.shape[2]
		image = torch.flip(image, dims=[1])  # TD TOP: row 0 = bottom, flip to image Y-down
		if src_w <= 0 or src_h <= 0:
			image = image.unsqueeze(0)
			if image.max() > 1.0:
				image = image.clamp(0.0, 255.0) / 255.0
			else:
				image = image.clamp(0.0, 1.0)
			return image
		ratio = min(self.engine_w / src_w, self.engine_h / src_h)
		new_w = round(src_w * ratio)
		new_h = round(src_h * ratio)
		image = F.interpolate(
			image.unsqueeze(0),
			size=(new_h, new_w),
			mode='bilinear',
			align_corners=False,
		)
		dw = self.engine_w - new_w
		dh = self.engine_h - new_h
		pad_left = round((dw / 2) - 0.1)
		pad_right = dw - pad_left
		pad_top = round((dh / 2) - 0.1)
		pad_bottom = dh - pad_top
		image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=114.0 / 255.0)
		if image.max() > 1.0:
			image = image.clamp(0.0, 255.0) / 255.0
		else:
			image = image.clamp(0.0, 1.0)
		return image

	def update_image(self):
		"""Get TOP as tensor and preprocess to engine size (like working ExampleExt)."""
		self.to_tensor.update(self.stream.cuda_stream)
		image = torch.as_tensor(self.to_tensor, device=self.device)
		return self.preprocess_image(image)

	def _draw_detections(self, img_np, dets):
		if cv2 is None or img_np is None:
			return img_np
		img = img_np.copy()
		if img.dtype != np.uint8 or img.max() <= 1.0:
			img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[-1] == 4:
			img = img[:, :, :3]
		for xc, yc, w, h, conf, cls_id in dets:
			x1 = int(xc - w / 2)
			y1 = int(yc - h / 2)
			x2 = int(xc + w / 2)
			y2 = int(yc + h / 2)
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			label = f"c{cls_id} {conf:.2f}"
			cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		return img

	def run(self, scriptOp):
		try:
			inference_on = bool(self.ownerComp.par.Inference.eval())
		except Exception:
			inference_on = True
		if not inference_on:
			return
		if self.model is None or self.ownerComp.par.Enginefile.val == '' or self.ownerComp.par.Venvpath.val == '':
			return
		if self.source is None or self.to_tensor is None or self.stream is None:
			return
		# Tensor from TOP, letterbox to engine size, then predict
		self.to_tensor.update(self.stream.cuda_stream)
		image = torch.as_tensor(self.to_tensor, device=self.device)
		if not torch.isfinite(image).all():
			image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
		orig_h, orig_w = int(image.shape[1]), int(image.shape[2])
		script_out = None  # segment mask or None -> pass original
		image_tensor = self.preprocess_image(image)
		image_tensor = image_tensor[:, [2, 1, 0], :, :].clone()  # BGR -> RGB

		try:
			results = self.model.predict(
				source=image_tensor,
				verbose=False,
				save=False,
				conf=self.conf_threshold,
				iou=self.iou_threshold,
			)
			torch.cuda.synchronize()
		except Exception as e:
			debug(e)
			return

		# Parse results: map from model (letterbox 640x640) to original, output in pixels
		try:
			tbl = op('table1')
			tbl.clear()
			tbl.appendRow(['det_id', 'class', 'confidence', 'x', 'y', 'width', 'height'])
		except Exception:
			tbl = None
		try:
			tbl_kpt = op('table2')
		except Exception:
			tbl_kpt = None

		if results and len(results) > 0 and results[0].boxes is not None and tbl is not None:
			from ultralytics.utils import ops
			boxes = results[0].boxes
			keypoints = getattr(results[0], 'keypoints', None)
			try:
				in1_top = op('in1')
				orig_h = int(in1_top.height)
				orig_w = int(in1_top.width)
			except Exception:
				return
			img0_shape = (orig_h, orig_w)
			img1_shape = (self.engine_h, self.engine_w)
			xyxy = boxes.xyxy
			if hasattr(xyxy, 'cpu'):
				xyxy = xyxy.cpu().numpy()
			xyxy = np.asarray(xyxy, dtype=np.float32)
			if xyxy.size == 0:
				xyxy_orig = xyxy
			else:
				xyxy_orig = ops.scale_boxes(img1_shape, xyxy.copy(), img0_shape)
			for i in range(len(xyxy_orig)):
				x1, y1, x2, y2 = float(xyxy_orig[i, 0]), float(xyxy_orig[i, 1]), float(xyxy_orig[i, 2]), float(xyxy_orig[i, 3])
				x_px = (x1 + x2) / 2
				y_px = orig_h - (y1 + y2) / 2  # image Y-down -> TD Y-up
				w_px = x2 - x1
				h_px = y2 - y1
				score = boxes.conf[i].item()
				cls_id = int(boxes.cls[i].item())
				tbl.appendRow([f"det_{i}", cls_id, score, x_px, y_px, w_px, h_px])

			# Pose model: add keypoints to table2 (det_id, kpt_id, x, y, visibility)
			# Use scale_coords like yolo_pose_test.py — correct letterbox handling for points
			if tbl_kpt is not None and keypoints is not None and len(keypoints) > 0:
				tbl_kpt.clear()
				tbl_kpt.appendRow(['det_id', 'kpt_id', 'x', 'y', 'visibility'])
				kpt_data = keypoints.data
				if hasattr(kpt_data, 'cpu'):
					kpt_data = kpt_data.cpu().numpy()
				kpt_data = np.asarray(kpt_data, dtype=np.float32)
				kpt_xy = kpt_data[..., :2].copy()
				ops.scale_coords(img1_shape, kpt_xy, img0_shape, padding=True)
				kpt_conf = self.kpt_conf_threshold
				for i in range(kpt_xy.shape[0]):
					for k in range(kpt_xy.shape[1]):
						v = float(kpt_data[i, k, 2]) if kpt_data.shape[-1] >= 3 else 1.0
						if v < kpt_conf:
							continue
						x_px = float(kpt_xy[i, k, 0])
						y_px = orig_h - float(kpt_xy[i, k, 1])  # image Y-down -> TD Y-up
						tbl_kpt.appendRow([f"det_{i}", k, x_px, y_px, v])

			# Script TOP: segment mask (task=segment) or pass-through original (else)
			masks = getattr(results[0], 'masks', None)
			try:
				task = str(self.ownerComp.par.Task.eval() or 'detect').strip().lower()
			except Exception:
				task = 'detect'
			if task == 'segment':
				try:
					if masks is not None and masks.data is not None and masks.data.shape[0] > 0:
						from ultralytics.utils.plotting import Colors
						colors = Colors()
						md = masks.data
						if hasattr(md, 'cpu'):
							md = md.cpu()
						md = md.float().to(device=self.device)
						ratio = min(self.engine_w / orig_w, self.engine_h / orig_h)
						new_w = round(orig_w * ratio)
						new_h = round(orig_h * ratio)
						pad_left = round((self.engine_w - new_w) / 2 - 0.1)
						pad_top = round((self.engine_h - new_h) / 2 - 0.1)
						md = md[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w]
						md = F.interpolate(md.unsqueeze(1), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(1)
						seg_out = torch.zeros(4, orig_h, orig_w, device=self.device, dtype=torch.float32)
						mask_thresh = 0.5
						for i in range(md.shape[0]):
							mi = (md[i] > mask_thresh).float()
							r, g, b = colors(i, bgr=False)
							seg_out[0] = seg_out[0] * (1 - mi) + (r / 255.0) * mi
							seg_out[1] = seg_out[1] * (1 - mi) + (g / 255.0) * mi
							seg_out[2] = seg_out[2] * (1 - mi) + (b / 255.0) * mi
						seg_out[3] = 1.0
					else:
						seg_out = torch.zeros(4, orig_h, orig_w, device=self.device, dtype=torch.float32)
						seg_out[3] = 1.0
					script_out = seg_out.permute(1, 2, 0).flip(0).contiguous()  # (C,H,W)->(H,W,C), flip Y for TD
				except Exception as e:
					debug(e)

		# Script TOP: segment mask or pass-through original
		try:
			if script_out is not None:
				out_iface = TopCUDAInterface(orig_w, orig_h, 4, np.float32)
				op('script1').copyCUDAMemory(script_out.data_ptr(), out_iface.size, out_iface.mem_shape)
			else:
				img = image[:4, :, :].float()
				if img.shape[0] == 3:
					alpha = torch.ones(1, orig_h, orig_w, device=self.device, dtype=torch.float32)
					img = torch.cat([img, alpha], dim=0)
				if img.max() > 1.0:
					img = img.clamp(0, 255) / 255.0
				else:
					img = img.clamp(0, 1)
				img_hwc = img.permute(1, 2, 0).contiguous()  # no flip: input already TD Y-up
				out_iface = TopCUDAInterface(orig_w, orig_h, 4, np.float32)
				op('script1').copyCUDAMemory(img_hwc.data_ptr(), out_iface.size, out_iface.mem_shape)
		except Exception as e:
			debug(e)

	def about(self, endpoint):
		if endpoint == 'Urlg':
			webbrowser.open('https://github.com/ultralytics/ultralytics', new=2)
		if endpoint == 'Urld':
			webbrowser.open('https://discord.gg/wNW8xkEjrf', new=2)
		if endpoint == 'Urlt':
			webbrowser.open('https://www.youtube.com/vjschool', new=2)
		if endpoint == 'Urla':
			webbrowser.open('https://olegcho.mp/', new=2)
		if endpoint == 'Urldonate':
			webbrowser.open('https://boosty.to/vjschool/donate', new=2)


class TopCUDAInterface:
	def __init__(self, width, height, num_comps, dtype):
		self.mem_shape = CUDAMemoryShape()
		self.mem_shape.width = width
		self.mem_shape.height = height
		self.mem_shape.numComps = num_comps
		self.mem_shape.dataType = dtype
		self.bytes_per_comp = np.dtype(dtype).itemsize
		self.size = width * height * num_comps * self.bytes_per_comp


class TopArrayInterface:
	def __init__(self, top, stream=0):
		self.top = top
		mem = top.cudaMemory(stream=stream)
		self.w, self.h = mem.shape.width, mem.shape.height
		self.num_comps = mem.shape.numComps
		self.dtype = mem.shape.dataType
		shape = (mem.shape.numComps, self.h, self.w)
		dtype_info = {'descr': [('', '<f4')], 'num_bytes': 4}
		dtype_descr = dtype_info['descr']
		num_bytes = dtype_info['num_bytes']
		num_bytes_px = num_bytes * mem.shape.numComps
		self.__cuda_array_interface__ = {
			"version": 3,
			"shape": shape,
			"typestr": dtype_descr[0][1],
			"descr": dtype_descr,
			"stream": stream,
			"strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
			"data": (mem.ptr, False),
		}

	@property
	def shape(self):
		return self.__cuda_array_interface__['shape']

	def update(self, stream=0):
		mem = self.top.cudaMemory(stream=stream)
		self.__cuda_array_interface__['stream'] = stream
		self.__cuda_array_interface__['data'] = (mem.ptr, False)
		return
