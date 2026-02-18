import os
import sys
from pathlib import Path
from datetime import datetime


class sdExt:
	"""
	StreamDiffusion client for TouchDesigner.
	Loads config.yaml, creates pipeline, runs img2img inference.
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.device = "cuda"
		self.stream = None
		self.config = None
		self.rgba_tensor = None
		self.output_interface = None
		self.n_controlnets = 0
		self.controlnet_enabled = False
		self.ipadapter_enabled = False
		self._has_ipadapter = False
		self.ipadapter_style_key = "ipadapter_main"

		# Путь к репо (Venvpath = корень проекта с StreamDiffusion и diffusers_ipadapter)
		venv_path = Path(parent().par.Venvpath.val)
		self.repo_root = venv_path
		# Сначала корень репо — чтобы использовался vendored diffusers_ipadapter
		if str(venv_path) not in sys.path:
			sys.path.insert(0, str(venv_path))
		sd_src = venv_path / "StreamDiffusion" / "src"
		if str(sd_src) not in sys.path:
			sys.path.insert(0, str(sd_src))

		import numpy as np
		import torch
		import yaml
		self.np = np
		self.torch = torch
		self.yaml = yaml

	# ── Config & Pipeline ────────────────────────────────────

	def load_config(self, config_path: str):
		"""Load YAML config and create pipeline."""
		try:
			config_path = Path(config_path)
			if not config_path.is_absolute():
				config_path = self.repo_root / config_path

			if not config_path.exists():
				self.log("Error", f"Config not found: {config_path}")
				return

			if self.stream is not None:
				self.stream.cleanup_gpu_memory()
				self.stream = None
			self.torch.cuda.empty_cache()

			with open(config_path, "r", encoding="utf-8") as f:
				self.config = self.yaml.safe_load(f)

			self.config["compile_engines_only"] = False
			self.config["build_engines_if_missing"] = False
			self.config["output_type"] = "pt"  # GPU tensor, без конвертации в PIL

			# engine_dir относительный — делаем абсолютным от корня проекта
			engine_dir = self.config.get("engine_dir", "engines")
			if not Path(engine_dir).is_absolute():
				self.config["engine_dir"] = str(self.repo_root / engine_dir)

			self.log("Status", f"Config loaded: {config_path.name}")
			self._create_pipeline()
		except Exception as e:
			self.log("Error", e)

	def _create_pipeline(self):
		"""Create StreamDiffusion wrapper from loaded config."""
		try:
			if self.config is None:
				self.log("Error", "No config loaded")
				return

			from streamdiffusion.config import create_wrapper_from_config
			self.stream = create_wrapper_from_config(self.config)

			self.num_inference_steps = self.config.get("num_inference_steps", 50)
			self.n_denoise_steps = len(self.config.get("t_index_list", [0]))
			self.n_controlnets = len(self.config.get("controlnets", []))
			self.controlnet_enabled = self.config.get("use_controlnet", False) and self.n_controlnets > 0
			self._has_ipadapter = bool(
				getattr(self.stream, "use_ipadapter", False)
				and (self.config.get("ipadapter_config") or self.config.get("ipadapters"))
			)
			self.ipadapter_enabled = self._has_ipadapter

			w = self.config.get("width", 512)
			h = self.config.get("height", 512)
			self._update_size(w, h)

			status = f"Pipeline ready: {w}x{h}, CN: {self.n_controlnets}"
			if self.ipadapter_enabled:
				status += ", IP-Adapter: on"
			self.log("Status", status)
		except Exception as e:
			self.log("Error", e)

	def _update_size(self, width: int, height: int):
		"""Update output tensors to match resolution."""
		self.rgba_tensor = self.torch.zeros((height, width, 4), dtype=self.torch.float32, device=self.device)
		self.rgba_tensor[..., 3] = 1.0
		self.output_interface = TopCUDAInterface(width, height, 4, self.np.float32)

	# ── Inference ────────────────────────────────────────────

	def generate(self, scriptOp):
		"""Called every frame by Script TOP."""
		if self.stream is None:
			return

		source = op("null1")
		if source is None or getattr(source, "width", 0) <= 0 or getattr(source, "height", 0) <= 0:
			return

		try:
			cuda_stream = self.torch.cuda.current_stream(self.device)
			to_tensor = TopArrayInterface(source)
			to_tensor.update(cuda_stream.cuda_stream)
			image = self.torch.as_tensor(to_tensor, device=self.device)
			image_tensor = self._preprocess(image)
		except Exception:
			return

		if self.controlnet_enabled:
			for i in range(self.n_controlnets):
				cn_top = op(f"cn{i}")
				if cn_top is None:
					continue
				try:
					cn_iface = TopArrayInterface(cn_top)
					cn_iface.update(cuda_stream.cuda_stream)
					cn_img = self.torch.as_tensor(cn_iface, device=self.device)
					cn_tensor = self._preprocess(cn_img)
					if self.torch.isnan(cn_tensor).any():
						continue
					self.stream.update_control_image(index=i, image=cn_tensor)
				except Exception:
					continue

		if self.ipadapter_enabled:
			ip_top = op("ip0")
			if ip_top is not None and getattr(ip_top, "width", 0) > 0 and getattr(ip_top, "height", 0) > 0:
				try:
					ip_iface = TopArrayInterface(ip_top)
					ip_iface.update(cuda_stream.cuda_stream)
					ip_img = self.torch.as_tensor(ip_iface, device=self.device)
					ip_tensor = self._preprocess(ip_img)
					if not self.torch.isnan(ip_tensor).any():
						self.stream.update_style_image(
							ip_tensor,
							is_stream=True,
							style_key=self.ipadapter_style_key,
						)
				except Exception:
					pass

		try:
			output_image = self.stream(image=image_tensor)
			output_tensor = self._postprocess(output_image)
			scriptOp.copyCUDAMemory(
				output_tensor.data_ptr(),
				self.output_interface.size,
				self.output_interface.mem_shape,
			)
		except Exception:
			pass

	def _preprocess(self, image):
		"""TOP tensor → pipeline input."""
		image = self.torch.flip(image, [1])
		image = self.torch.clamp(image, 0, 1)
		image = image[:3, :, :]
		return image.unsqueeze(0)

	def _postprocess(self, image):
		"""Pipeline output (B,C,H,W) or (C,H,W) → RGBA for TOP."""
		if image.dim() == 4:
			image = image.squeeze(0)  # (B,C,H,W) → (C,H,W)
		image = image.clamp(0, 1)
		image = self.torch.flip(image, [1])
		image = image.permute(1, 2, 0)  # (C,H,W) → (H,W,C)
		self.rgba_tensor[..., :3] = image
		return self.rgba_tensor

	# ── Runtime parameter updates ────────────────────────────

	def _collect_prompts(self):
		"""Собираем все непустые блоки из sequence Prompts."""
		seq = parent().seq.Prompts
		prompt_list = []
		for block in seq:
			text = str(block.par.Prompt.val).strip()
			if text:
				w = float(block.par.Weight.val)
				prompt_list.append((text, w))
		return prompt_list

	def update_prompts(self):
		"""Полное обновление промптов (текст + веса, перекодирует)."""
		if self.stream is None:
			return
		prompt_list = self._collect_prompts()
		if not prompt_list:
			return
		if len(prompt_list) == 1:
			self.stream.update_prompt(prompt_list[0][0])
		else:
			self.stream.update_prompt(prompt_list)

	def update_prompt_weights(self):
		"""Обновляем только веса (без перекодирования промптов)."""
		if self.stream is None:
			return
		prompt_list = self._collect_prompts()
		weights = [w for _, w in prompt_list]
		if weights and hasattr(self.stream, 'stream'):
			try:
				self.stream.stream._param_updater.update_prompt_weights(weights)
			except Exception:
				self.update_prompts()

	def update_guidance(self, guidance_scale: float, delta: float):
		if self.stream is not None:
			self.stream.update_stream_params(
				guidance_scale=guidance_scale,
				delta=delta,
			)

	def update_seed(self, seed: int):
		if self.stream is not None:
			self.stream.update_stream_params(seed=seed)

	def update_controlnet_enabled(self, enabled: bool):
		self.controlnet_enabled = enabled and self.n_controlnets > 0

	def update_controlnet_scale(self, index: int, scale: float):
		if self.stream is not None and index < self.n_controlnets:
			self.stream.stream._controlnet_module.update_controlnet_scale(index, scale)

	def update_ipadapter_enabled(self, enabled: bool):
		self.ipadapter_enabled = enabled and getattr(self, "_has_ipadapter", False)

	def update_ipadapter_scale(self, scale: float):
		if self.stream is None or not getattr(self.stream, "use_ipadapter", False):
			return
		try:
			self.stream.update_stream_params(ipadapter_config={"scale": float(scale)})
		except Exception:
			pass

	def update_safety_checker(self, enabled: bool):
		if self.stream is not None:
			try:
				self.stream.update_stream_params(use_safety_checker=enabled)
			except Exception:
				pass

	def update_denoise(self, strength: float):
		"""Пересчёт t_index_list из float 0‑1.
		0 = оригинал (индексы у конца), 1 = макс. денойз (индексы от начала)."""
		if self.stream is None:
			return
		strength = max(0.0, min(1.0, strength))
		n = self.n_denoise_steps
		last_idx = self.num_inference_steps - 1
		start_idx = int(round((1.0 - strength) * last_idx))
		if n == 1:
			t_list = [start_idx]
		else:
			span = last_idx - start_idx
			t_list = [int(round(start_idx + i * span / (n - 1))) for i in range(n)]
		self.stream.update_stream_params(t_index_list=t_list)

	def update_sampler(self, sampler: str):
		if self.stream is not None:
			self.stream.stream.set_scheduler(sampler=sampler)

	# ── TouchDesigner callbacks ──────────────────────────────

	def parexec_onPulse(self, par):
		if par.name == "Loadmodel":
			model_name = parent().par.Model.val
			table = op("null2")
			config_path = None
			for row in range(table.numRows):
				if table[row, 0].val == model_name:
					config_path = table[row, 1].val
					break
			if config_path:
				self.load_config(config_path)
			else:
				self.log("Error", f"Config not found for: {model_name}")

	def parexec_onValueChange(self, par, prev):
		if self.stream is None:
			return
		name = par.name
		if name.endswith("prompt") and name.startswith("Prompts"):
			self.update_prompts()
		elif name.endswith("weight") and name.startswith("Prompts"):
			self.update_prompt_weights()
		elif name == "Cfgscale":
			self.update_guidance(par.val, parent().par.Deltamult.val)
		elif name == "Deltamult":
			self.update_guidance(parent().par.Cfgscale.val, par.val)
		elif name == "Seed":
			self.update_seed(int(par.val))
		elif name == "Denoise":
			self.update_denoise(par.val)
		elif name == "Sampler":
			self.update_sampler(str(par.val))
		elif name == "Controlnet":
			self.update_controlnet_enabled(bool(par.val))
		elif name.startswith("Controlnet") and name.endswith("weight"):
			idx = int(name[len("Controlnet"):-len("weight")]) - 1
			self.update_controlnet_scale(idx, float(par.val))
		elif name == "Ipadapter":
			self.update_ipadapter_enabled(bool(par.val))
		elif name == "Ipadapterweight":
			self.update_ipadapter_scale(float(par.val))
		elif name == "Safety":
			self.update_safety_checker(bool(par.val))

	# ── Utils ────────────────────────────────────────────────

	def log(self, status, message):
		current_time = datetime.now().strftime("%H:%M:%S")
		fifo = op("fifo1")
		if fifo:
			fifo.appendRow([current_time, status, message])
		print(f"[{current_time}] {status}: {message}")


class TopCUDAInterface:
	def __init__(self, width, height, num_comps, dtype):
		import numpy as np
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
		shape = (mem.shape.numComps, self.h, self.w)
		num_bytes = 4  # float32
		num_bytes_px = num_bytes * mem.shape.numComps

		self.__cuda_array_interface__ = {
			"version": 3,
			"shape": shape,
			"typestr": "<f4",
			"descr": [("", "<f4")],
			"stream": stream,
			"strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
			"data": (mem.ptr, False),
		}

	def update(self, stream=0):
		mem = self.top.cudaMemory(stream=stream)
		self.__cuda_array_interface__["stream"] = stream
		self.__cuda_array_interface__["data"] = (mem.ptr, False)
