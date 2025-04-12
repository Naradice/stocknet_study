import csv
import json
import os

import numpy as np
import pandas as pd
import torch
from datetime import datetime

class Logger:

  @classmethod
  def connect_drive(cls, mount_path='/content/drive'):
    from google.colab import drive
    drive.mount(mount_path)

  def __init__(self, model_name, version, base_path=None, storage_handler='colab', max_retry=3, local_cache_period=10, client_id=None):
    """ Logging class to store training logs

    Args:
        model_name (str): It create a folder {base_path}/{model_name}/.
        verison (str): It create a file {base_path}/{model_name}/{model_name}_v{version}.csv.
        base_path (str, optional): Base path to store logs. If you use cloud storage, this is used as temporal folder. Defaults to None.
        storage_handler (str|BaseHandler, optional): It change storage service. 'colab' can be selected. Defaults to 'colab'.
        max_retry (int, optional): max count of retry when store logs via network. Defaults to 3.
        local_cache_period(int, optional): Valid for cloud storage only. period to chache logs until send it to the storage. Defaults to 10.
        client_id(str, optional): client_id to authenticate cloud service with OAuth2.0/OIDC. Defaults to None.
    """
    # define common veriables
    MOUNT_PATH = '/content/drive'
    self.__use_cloud_storage = False
    self.__init_storage = lambda : None
    self.__local_cache_period = local_cache_period
    self.model_name = model_name
    self.version = version
    self.max_retry = max_retry

    # define variables depends on env
    if storage_handler == 'colab':
      # this case we store logs on mounted path
      self.__init_colab()
      self.__init_storage = self.__init_colab
      if base_path is None:
        self.base_path = MOUNT_PATH
      else:
        base_pathes = [p for p in base_path.split('/') if len(p) > 0]
        self.base_path = os.path.join(MOUNT_PATH, 'My Drive', *base_pathes)
    elif type(storage_handler) is str:
      raise ValueError(f"{storage_handler} is not supported. Please create StorageHandler for the service.")
    elif storage_handler is not None:
      # this case we store logs on app folder of dropbox, using cloud_storage_handlder
      self.__cloud_handler = storage_handler
      if self.__cloud_handler.refresh_token is None:
        self.__cloud_handler.authenticate()
      self.__use_cloud_storage = True
      if base_path is None:
        self.base_path = './'
      else:
        self.base_path = base_path
    else:
      if base_path is None:
        self.base_path = './'
      else:
        self.base_path = base_path
    model_log_folder = os.path.join(self.base_path, model_name)
    if not os.path.exists(model_log_folder):
        os.makedirs(model_log_folder)
    file_name = f"{model_name}_v{version}.csv"
    self.log_file_path = os.path.join(model_log_folder, file_name)
    self.__cache = []

  def __init_colab(self):
    from google.colab import drive
    drive.mount(MOUNT_PATH)

  def __store_files_to_cloud_storage(self, file_path):
    try:
      self.__cloud_handler.upload_training_results(self.model_name, [file_path])
    except Exception as e:
      print(f"failed to save logs to dropbox: {e}")

  def reset(self, model_name=None, file_name=None):
    if file_name is None:
      file_name = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if model_name is None:
      if file_name is None:
        raise ValueError("Either model_name or file_name should be specified")
      self.log_file_path = os.path.join(self.base_path, file_name)
    else:
      model_log_folder = os.path.join(self.base_path, model_name)
      if not os.path.exists(model_log_folder):
        os.makedirs(model_log_folder)
      self.log_file_path = os.path.join(model_log_folder, file_name)
    self.__cache = []

  def __cache_log(self, log_entry: list):
    self.__cache.append(log_entry)

  def __append_log(self, log_entry:list, retry_count=0):
      try:
          with open(self.log_file_path, 'a') as log_file:
            writer = csv.writer(log_file)
            if len(self.__cache) > 0:
              writer.writerows(self.__cache)
              self.__cache = []
            writer.writerow(log_entry)
      except Exception as e:
        if retry_count < self.max_retry:
          if retry_count == 0:
            print(e)
          self.__init_storage()
          self.__append_log(log_entry, retry_count+1)
        else:
          self.__cache.append(log_entry)

  def save_params(self, params:dict, model_name=None, model_version=None):
    data_folder = os.path.dirname(self.log_file_path)
    if model_name is None:
      model_name = self.model_name
    if model_version is None:
      model_version = self.version
    if not os.path.exists(data_folder):
      os.makedirs(data_folder)
    param_file_path = os.path.join(data_folder, f'{model_name}_v{model_version}_params.json')
    with open(param_file_path, mode="w") as fp:
      json.dump(params, fp)
    if self.__use_cloud_storage:
      self.__store_files_to_cloud_storage(param_file_path)

  def save_model(self, model, model_name=None, model_version=None):
    if model is not None:
      data_folder = os.path.dirname(self.log_file_path)
      if model_name is None:
        model_name = self.model_name
      if model_version is None:
        model_version = self.version
      if not os.path.exists(data_folder):
        os.makedirs(data_folder)
      param_file_path = os.path.join(data_folder, f'{model_name}_v{model_version}.torch')
      torch.save(model.state_dict(), param_file_path)
      if self.__use_cloud_storage:
        self.__store_files_to_cloud_storage(param_file_path)

  def save_checkpoint(self, model, optimizer, scheduler, model_name, model_version, **kwargs):
    if model is not None:
      data_folder = os.path.dirname(self.log_file_path)
      model_path = os.path.join(data_folder, f'{model_name}_v{model_version}.torch')
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        **kwargs
      }, model_path)
      if self.__use_cloud_storage:
        self.__store_files_to_cloud_storage(model_path)

  def save_logs(self):
    if len(self.__cache) > 0:
      with open(self.log_file_path, 'a') as log_file:
        if len(self.__cache) > 0:
          writer = csv.writer(log_file)
          writer.writerows(self.__cache)
    if self.__use_cloud_storage:
        self.__store_files_to_cloud_storage(self.log_file_path)

  def add_training_log(self, training_loss, validation_loss, log_entry:list=None):
    timestamp = datetime.now().isoformat()
    basic_entry = [timestamp, training_loss, validation_loss]
    if log_entry is not None:
      if type(log_entry) is list and len(log_entry) > 0:
        basic_entry.extend(log_entry)
    if len(self.__cache) < self.__local_cache_period:
      self.__cache_log(basic_entry)
    else:
      self.__append_log(basic_entry)
      if self.__use_cloud_storage:
        self.__store_files_to_cloud_storage(self.log_file_path)

  def get_min_losses(self, train_loss_column=1, val_loss_column=2):
    logs = None
    if os.path.exists(self.log_file_path) is False:
      file_name = os.path.dirname(self.log_file_path)
      destination_path = f'/{self.model_name}/{file_name}'
      response = self.__cloud_handler.download_file(destination_path, self.log_file_path)
      if response is not None:
        logs = pd.read_csv(self.log_file_path)
    else:
      logs = pd.read_csv(self.log_file_path)

    if logs is None:
      print("no log available")
      return np.inf, np.inf
    else:
      if type(train_loss_column) is int:
        train_loss = logs.iloc[:, train_loss_column]
      elif type(train_loss_column) is str:
        train_loss = logs[train_loss_column]
      min_train_loss = train_loss.min()

      if type(val_loss_column) is int:
        val_loss = logs.iloc[:, val_loss_column]
      elif type(val_loss_column) is str:
        val_loss = logs[val_loss_column]
      min_val_loss = val_loss.min()

      return min_train_loss, min_val_loss