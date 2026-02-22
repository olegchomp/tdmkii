'''
Flux Klein 4B — TouchDesigner extension for image editing via TRT engines.
Load pipeline by Pulse par "Loadengine". Inference runs when par Inference = True.
Input: null1 (image), Output: scriptOp (Script TOP).
Params: Venvpath, Configfile, Loadengine, Inference, Prompt, Steps (1-8, default 4).
'''
import os
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import webbrowser


class FluxKleinExt:
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self._set_dimensions('')
		self.device = 'cuda'
		self.pipe = None
		self.stream = None
		self.trt_engines = {}
		self.w = 512
		self.h = 512
		self.rgba_tensor = None
		self.output_interface = None
		self.to_tensor = None
		self.source = None
		self.inference_on = True
		self.prompt = 'make it sunset'
		self._cached_prompt = None
		self._cached_prompt_embeds = None
		try:
			self.source = op('null1')
			self.to_tensor = TopArrayInterface(self.source)
		except Exception:
			pass

	def Loadengine(self):
		"""Unload old pipeline, load config and create Flux Klein pipeline. Pulse par Loadengine triggers this."""
		if self.pipe is not None:
			del self.pipe
			self.pipe = None
		for k in list(self.trt_engines.keys()):
			del self.trt_engines[k]
		self.trt_engines.clear()
		self.stream = None
		self._cached_prompt = None
		self._cached_prompt_embeds = None
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		try:
			self.source = op('null1')
			self.to_tensor = TopArrayInterface(self.source)
		except Exception:
			self.source = None
			self.to_tensor = None

		venv_path = self.ownerComp.par.Venvpath.val or ""
		config_file = self.ownerComp.par.Configfile.val or ""
		if not venv_path or not config_file:
			self._set_dimensions('')
			return

		repo = Path(venv_path)
		config_path = repo / config_file if not Path(config_file).is_absolute() else Path(config_file)
		if not config_path.exists():
			debug(f"Config not found: {config_path}")
			return

		if str(repo) not in sys.path:
			sys.path.insert(0, str(repo))

		try:
			import yaml
			import tensorrt as trt
			with open(config_path) as f:
				cfg = yaml.safe_load(f)

			self.w = cfg.get('width', 512)
			self.h = cfg.get('height', 512)
			tr_path = Path(cfg['transformer_engine'])
			vae_dec_path = Path(cfg['vae_engine'])
			vae_enc_path = Path(cfg['vae_encoder_engine']) if cfg.get('vae_encoder_engine') else None
			if not tr_path.is_absolute():
				tr_path = repo / tr_path
			if not vae_dec_path.is_absolute():
				vae_dec_path = repo / vae_dec_path
			if vae_enc_path and not Path(vae_enc_path).is_absolute():
				vae_enc_path = repo / vae_enc_path

			if not tr_path.exists() or not vae_dec_path.exists():
				debug(f"Engine files not found")
				return

			TrtEngine = _make_trt_engine_class(trt)
			transformer_trt = TrtEngine(str(tr_path), 'transformer')
			vae_dec_trt = TrtEngine(str(vae_dec_path), 'VAE decoder')
			vae_enc_trt = TrtEngine(str(vae_enc_path), 'VAE encoder') if vae_enc_path and vae_enc_path.exists() else None
			self.trt_engines['transformer'] = transformer_trt
			self.trt_engines['vae_dec'] = vae_dec_trt
			if vae_enc_trt:
				self.trt_engines['vae_enc'] = vae_enc_trt

			from diffusers_flux2 import Flux2KleinPipeline
			self.pipe = Flux2KleinPipeline.from_pretrained(
				cfg.get('model_id', 'black-forest-labs/FLUX.2-klein-4B'),
				torch_dtype=torch.bfloat16,
			)
			self.pipe.to(self.device)
			self.pipe.transformer.to('cpu')
			self.pipe.vae.to('cpu')
			gc.collect()
			torch.cuda.empty_cache()
			self.pipe.vae.to(self.device)

			_patch_transformer(self.pipe, transformer_trt)
			if vae_enc_trt:
				_patch_vae_encoder(self.pipe, vae_enc_trt)
			_patch_vae_decoder(self.pipe, vae_dec_trt)

			self.stream = torch.cuda.current_stream(device=self.device)
			self.rgba_tensor = torch.zeros((self.h, self.w, 4), dtype=torch.float32, device=self.device)
			self.rgba_tensor[..., 3] = 1.0
			self.output_interface = TopCUDAInterface(self.w, self.h, 4, np.float32)
			self._set_dimensions(f"{self.w}x{self.h}")
			try:
				self.inference_on = bool(self.ownerComp.par.Inference.eval())
			except Exception:
				pass
			try:
				self.prompt = str(self.ownerComp.par.Prompt.val or 'make it sunset').strip() or 'make it sunset'
			except Exception:
				self.prompt = 'make it sunset'
			try:
				op('fit2').par.resolutionw = self.w
				op('fit2').par.resolutionh = self.h
			except Exception:
				pass
		except Exception as e:
			debug(e)
			self.pipe = None
			self._set_dimensions('')

	def parexec_onPulse(self, par):
		if par.name == 'Loadengine':
			self.Loadengine()

	def _set_dimensions(self, val):
		try:
			if hasattr(self.ownerComp.par, 'Dimensions'):
				self.ownerComp.par.Dimensions = val
		except Exception:
			pass

	def parexec_onValueChange(self, par, prev):
		if par.name == 'Inference':
			try:
				self.inference_on = bool(par.eval())
			except Exception:
				pass
		elif par.name == 'Prompt':
			try:
				self.prompt = str(par.val or 'make it sunset').strip() or 'make it sunset'
			except Exception:
				self.prompt = 'make it sunset'
		elif par.name == 'Steps':
			pass  # read in run()

	def _steps(self):
		try:
			return max(1, min(8, int(self.ownerComp.par.Steps.eval() or 4)))
		except (AttributeError, TypeError, ValueError):
			return 4

	def _preprocess(self, image):
		"""TOP tensor (C,H,W) -> pipeline input: (1,3,H,W), 0-1, RGB."""
		image = image[:3, :, :].float()
		if image.max() > 1.0:
			image = image.clamp(0, 255) / 255.0
		else:
			image = image.clamp(0, 1)
		image = torch.flip(image, [1])  # TD Y-up -> image Y-down
		return image.unsqueeze(0)

	def _postprocess(self, out_tensor):
		"""Pipeline output (1,C,H,W) or (C,H,W) -> RGBA for TOP, TD Y-up."""
		if out_tensor.dim() == 4:
			out_tensor = out_tensor.squeeze(0)
		out_tensor = out_tensor.clamp(0, 1)
		out_tensor = torch.flip(out_tensor, [1])  # image Y-down -> TD Y-up
		out_tensor = out_tensor.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
		self.rgba_tensor[..., :3] = out_tensor
		return self.rgba_tensor

	def run(self, scriptOp):
		if not self.inference_on or self.pipe is None:
			return
		if self.source is None or self.to_tensor is None or self.stream is None:
			return
		if self.ownerComp.par.Venvpath.val == '' or self.ownerComp.par.Configfile.val == '':
			return

		self.to_tensor.update(self.stream.cuda_stream)
		image = torch.as_tensor(self.to_tensor, device=self.device)
		if not torch.isfinite(image).all():
			image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
		image_tensor = self._preprocess(image)
		image_tensor = image_tensor.to(torch.bfloat16)
		if image_tensor.shape[2] != self.h or image_tensor.shape[3] != self.w:
			image_tensor = torch.nn.functional.interpolate(
				image_tensor.float(),
				size=(self.h, self.w),
				mode='bilinear',
				align_corners=False,
			).to(torch.bfloat16)

		try:
			if self._cached_prompt != self.prompt:
				self._cached_prompt = self.prompt
				with torch.inference_mode():
					self._cached_prompt_embeds, _ = self.pipe.encode_prompt(
						prompt=self.prompt,
						device=self.device,
						num_images_per_prompt=1,
					)
			with torch.inference_mode():
				result = self.pipe(
					image=image_tensor,
					prompt=None,
					prompt_embeds=self._cached_prompt_embeds,
					height=self.h,
					width=self.w,
					num_inference_steps=self._steps(),
					output_type='pt',
				)
			torch.cuda.synchronize()
			out_tensor = result.images[0].unsqueeze(0) if result.images[0].dim() == 3 else result.images[0]
			output = self._postprocess(out_tensor.float())
			scriptOp.copyCUDAMemory(output.data_ptr(), self.output_interface.size, self.output_interface.mem_shape)
		except Exception as e:
			debug(e)

	def about(self, endpoint):
		if endpoint == 'Urlg':
			webbrowser.open('https://github.com/black-forest-labs/flux-klein', new=2)
		if endpoint == 'Urld':
			webbrowser.open('https://discord.gg/wNW8xkEjrf', new=2)
		if endpoint == 'Urla':
			webbrowser.open('https://olegcho.mp/', new=2)


def _make_trt_engine_class(trt):
	class TrtEngine:
		def __init__(self, engine_path, label=''):
			self.logger = trt.Logger(trt.Logger.WARNING)
			runtime = trt.Runtime(self.logger)
			with open(engine_path, 'rb') as f:
				self.engine = runtime.deserialize_cuda_engine(f.read())
			self.context = self.engine.create_execution_context()
			self.stream = torch.cuda.Stream()
			self.io_info = {}
			for i in range(self.engine.num_io_tensors):
				name = self.engine.get_tensor_name(i)
				mode = self.engine.get_tensor_mode(name)
				dtype_trt = self.engine.get_tensor_dtype(name)
				dtype_torch = {
					trt.bfloat16: torch.bfloat16,
					trt.float16: torch.float16,
					trt.float32: torch.float32,
				}.get(dtype_trt, torch.float32)
				is_input = mode == trt.TensorIOMode.INPUT
				self.io_info[name] = {'dtype': dtype_torch, 'input': is_input}

		def run(self, inputs):
			device = 'cuda'
			for name, tensor in inputs.items():
				info = self.io_info[name]
				t = tensor.detach().to(dtype=info['dtype']).contiguous().cuda()
				self.context.set_tensor_address(name, t.data_ptr())
				inputs[name] = t
			outputs = {}
			for name, info in self.io_info.items():
				if not info['input']:
					shape = self.context.get_tensor_shape(name)
					t = torch.empty(list(shape), dtype=info['dtype'], device=device)
					self.context.set_tensor_address(name, t.data_ptr())
					outputs[name] = t
			with torch.cuda.stream(self.stream):
				self.context.execute_async_v3(self.stream.cuda_stream)
			self.stream.synchronize()
			return outputs
	return TrtEngine


class _FakeLatentDist:
	def __init__(self, latent): self._latent = latent
	def mode(self): return self._latent
	def sample(self, generator=None): return self._latent


def _patch_transformer(pipe, engine):
	def trt_forward(hidden_states, encoder_hidden_states=None, timestep=None, img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None, return_dict=True, **kwargs):
		result = engine.run({
			'hidden_states': hidden_states, 'encoder_hidden_states': encoder_hidden_states,
			'timestep': timestep, 'img_ids': img_ids, 'txt_ids': txt_ids,
		})
		sample = result['output'].to(dtype=hidden_states.dtype)
		if return_dict:
			from diffusers.models.modeling_outputs import Transformer2DModelOutput
			return Transformer2DModelOutput(sample=sample)
		return (sample,)
	pipe.transformer.forward = trt_forward


def _patch_vae_encoder(pipe, engine):
	def trt_encode(x, return_dict=True, **kwargs):
		result = engine.run({'image': x})
		latent = result['latent'].to(dtype=x.dtype)
		fake = _FakeLatentDist(latent)
		if return_dict:
			out = type('_Enc', (), {})()
			out.latent_dist = fake
			return out
		return (fake,)
	pipe.vae.encode = trt_encode


def _patch_vae_decoder(pipe, engine):
	def trt_decode(z, return_dict=True, **kwargs):
		result = engine.run({'latent': z})
		image = result['image'].to(dtype=z.dtype)
		if return_dict:
			try:
				from diffusers.models.autoencoders.vae import DecoderOutput
				return DecoderOutput(sample=image)
			except ImportError:
				out = type('_Dec', (), {})()
				out.sample = image
				return out
		return (image,)
	pipe.vae.decode = trt_decode


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
		shape = (mem.shape.numComps, self.h, self.w)
		num_bytes = 4
		num_bytes_px = num_bytes * mem.shape.numComps
		self.__cuda_array_interface__ = {
			'version': 3, 'shape': shape, 'typestr': '<f4', 'descr': [('', '<f4')],
			'stream': stream, 'strides': (num_bytes, num_bytes_px * self.w, num_bytes_px), 'data': (mem.ptr, False),
		}

	def update(self, stream=0):
		mem = self.top.cudaMemory(stream=stream)
		self.__cuda_array_interface__['stream'] = stream
		self.__cuda_array_interface__['data'] = (mem.ptr, False)
