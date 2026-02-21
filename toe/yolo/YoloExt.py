'''
YoloExt: YOLO inference in TouchDesigner via Ultralytics.
No auto-load: init only sets up component. Load engine by Pulse par "Loadengine" (parexec -> Loadengine()).
Inference runs only when par Inference = True. Fills table1 with detections.
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
			self.source = op('null1')
			self.to_tensor = TopArrayInterface(self.source)
		except Exception:
			self.source = None
			self.to_tensor = None

	def Loadengine(self):
		"""Load or reload engine via Ultralytics YOLO(engine_path). Same TRT as export."""
		self.model = None
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

	def preprocess_image(self, image):
		"""(C, H, W) -> (1, 3, engine_h, engine_w) normalized 0-1 for model.predict(source=tensor)."""
		image = image[:3, :, :]
		image = F.interpolate(
			image.unsqueeze(0),
			size=(self.engine_h, self.engine_w),
			mode='bilinear',
			align_corners=False,
		)
		# YOLO expects 0-1; TOP may be 0-255 or wrong range
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
		# Tensor from TOP, resize to engine size, then predict
		self.to_tensor.update(self.stream.cuda_stream)
		image = torch.as_tensor(self.to_tensor, device=self.device)
		image_tensor = self.preprocess_image(image)

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

		# Parse results and fill table1 (x, y, width, height from YOLO xywhn, normalized 0–1)
		try:
			tbl = op('table1')
			tbl.clear()
			tbl.appendRow(['det_id', 'class', 'confidence', 'x', 'y', 'width', 'height'])
		except Exception:
			tbl = None
		if results and len(results) > 0 and results[0].boxes is not None and tbl is not None:
			boxes = results[0].boxes
			xywhn = boxes.xywhn
			if hasattr(xywhn, 'cpu'):
				xywhn = xywhn.cpu().numpy()
			xywhn = np.asarray(xywhn)
			for i in range(len(xywhn)):
				x_norm, y_norm, w_norm, h_norm = float(xywhn[i, 0]), float(xywhn[i, 1]), float(xywhn[i, 2]), float(xywhn[i, 3])
				score = boxes.conf[i].item()
				cls_id = int(boxes.cls[i].item())
				tbl.appendRow([f"det_{i}", cls_id, score, x_norm, y_norm, w_norm, h_norm])

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
