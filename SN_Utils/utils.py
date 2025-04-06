import inspect
import json
import os
import torch

def has_param(func, param_name):
    sig = inspect.signature(func)
    return param_name in sig.parameters

def load_model(model_name, model_class, model_version, device, optimizer_class, scheduler_class, train=True, storage_handler=None, model_folder=None, lr=1e-3):
  def load_torch_model(model_path, device, params, optimizer=None, scheduler=None):
    if device is None:
      if torch.cuda.is_available():
        check_point = torch.load(model_path, weights_only=True)
      else:
        check_point = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    else:
      check_point = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in check_point:
      model.load_state_dict(check_point['model_state_dict'])
      if optimizer is not None:
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
      if scheduler is not None:
        scheduler.load_state_dict(check_point['scheduler_state_dict'])
      return params, model, optimizer, scheduler
    else:
      model.load_state_dict(check_point)
      return params, model, None, None

  if model_folder is None:
    model_folder = base_folder
  model_folder = os.path.join(model_folder, model_name)

  params_file_name = f'{model_folder}/{model_name}_v{model_version}_params.json'
  if os.path.exists(params_file_name) is False:
    if storage_handler is None:
      print(f"exsisting model params not found on {params_file_name}.")
      return None, None, None, None
    else:
      response = storage_handler.download_file(f"/{model_name}/{model_name}_v{model_version}_params.json", params_file_name)
      if response is None:
        print("exsisting model params not found.")
        return None, None, None, None
  with open(params_file_name) as fp:
      params = json.load(fp)
  if has_param(model_class, 'feature_size'):
    params["feature_size"] = len(params["features"])
  model = model_class(**params).to(device)
  optimizer = optimizer_class(model.parameters(), lr=lr)
  scheduler = scheduler_class(optimizer, 1.0)
  if train:
    model_path = f'{model_folder}/{model_name}_train_v{model_version}.torch'
  else:
    model_path = f'{model_folder}/{model_name}_v{model_version}.torch'
    
  if os.path.exists(model_path) is False:
    if storage_handler is None:
      print("exsisting model not found.")
      return None, None, None, None
    file_name = os.path.basename(model_path)
    response = storage_handler.download_file(f"/{model_name}/{file_name}", model_path)
    if response is None:
      print("exsisting model not found.")
      return None, None, None, None

  if optimizer_class is None:
    return load_torch_model(model_path, device, params)
  optimizer = optimizer_class(model.parameters(), lr=0.001)
  if scheduler_class is None:
    return load_torch_model(model_path, device, params, optimizer)
  scheduler = scheduler_class(optimizer, 1.0)
  return load_torch_model(model_path, device, params, optimizer, scheduler)