import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model_path, model, optimizer, scheduler, best_loss, **kwargs):
    directory = os.path.dirname(model_path)
    os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            **kwargs,
        },
        model_path,
    )
    print(f"model checkpoint saved at {model_path}")


def load_model_params(model_folder, model_name, model_version, storage_handler=None):
    default_response = None
    params_file_name = f"{model_folder}/{model_name}_v{model_version}_params.json"
    if os.path.exists(params_file_name) is False:
        if storage_handler is None:
            print(f"exsisting model params not found on {params_file_name}.")
            return default_response
        else:
            response = storage_handler.download_file(f"/{model_name}/{model_name}_v{model_version}_params.json", params_file_name)
            if response is None:
                print("exsisting model params not found.")
                return default_response
    with open(params_file_name) as fp:
        params = json.load(fp)
    return params


def load_model(create_model_func, model_folder, model_name, model_version, storage_handler=None, device=None):
    if device is None:
        device = get_device()
    default_response = None, None
    params = load_model_params(model_folder, model_name, model_version, storage_handler)
    if params is None:
        return default_response

    model = create_model_func(**params).to(device)
    return params, model


def load_model_checkpoint(
    create_model_func,
    model_name,
    model_version,
    model_folder,
    optimizer_class,
    scheduler_class,
    train=True,
    storage_handler=None,
    optimizer_kwargs={"lr": 1e-3},
    scheduler_kwargs={"step_size": 1, "gamma": 0.95},
):
    default_response = (False, None, None, None, None, np.inf)

    params, model = load_model(create_model_func, model_folder, model_name, model_version, storage_handler)
    if model is None:
        return default_response
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    scheduler = scheduler_class(optimizer, **scheduler_kwargs)
    if train:
        model_path = f"{model_folder}/{model_name}_train_v{model_version}.torch"
    else:
        model_path = f"{model_folder}/{model_name}_v{model_version}.torch"
    if os.path.exists(model_path) is False:
        if storage_handler is None:
            print("exsisting model not found.")
            return default_response
        else:
            file_name = os.path.basename(model_path)
            response = storage_handler.download_file(f"/{model_name}/{file_name}", model_path)
            if response is None:
                print("exsisting model not found.")
                return default_response

    if torch.cuda.is_available():
        check_point = torch.load(model_path)
    else:
        check_point = torch.load(model_path, map_location=torch.device("cpu"))
    if "model_state_dict" in check_point:
        model.load_state_dict(check_point["model_state_dict"])
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        scheduler.load_state_dict(check_point["scheduler_state_dict"])
        if "best_loss" in check_point:
            best_loss = check_point["best_loss"]
        else:
            print("best_loss not found.")
            best_loss = np.inf
        return params, model, optimizer, scheduler, best_loss
    else:
        print("checkpoint is not available.")
        model.load_state_dict(check_point)
        return params, model, None, None, np.inf


class Logger:
    @classmethod
    def connect_drive(cls, mount_path="/content/drive"):
        from google.colab import drive

        drive.mount(mount_path)

    def __init__(self, model_name, version, base_path=None, storage_handler="colab", max_retry=3, local_cache_period=10):
        """Logging class to store training logs

        Args:
            model_name (str): It create a folder {base_path}/{model_name}/.
            verison (str): It create a file {base_path}/{model_name}/{model_name}_v{version}.csv.
            base_path (str, optional): Base path to store logs. If you use cloud storage, this is used as temporal folder. Defaults to None.
            storage_handler (str|BaseHandler, optional): It change storage service. 'colab' can be selected. Defaults to 'colab'.
            max_retry (int, optional): max count of retry when store logs via network. Defaults to 3.
            local_cache_period(int, optional): Valid for cloud storage only. period to chache logs until send it to the storage. Defaults to 10.
        """
        # define common veriables
        self.MOUNT_PATH = "/content/drive"
        self.__use_cloud_storage = False
        self.__init_storage = lambda: None
        self.__local_cache_period = local_cache_period
        self.model_name = model_name
        self.version = version
        self.max_retry = max_retry

        # define variables depends on env
        if storage_handler == "colab":
            # this case we store logs on mounted path
            self.__init_colab()
            self.__init_storage = self.__init_colab
            if base_path is None:
                self.base_path = self.MOUNT_PATH
            else:
                base_pathes = [p for p in base_path.split("/") if len(p) > 0]
                self.base_path = os.path.join(self.MOUNT_PATH, "My Drive", *base_pathes)
        elif type(storage_handler) is str:
            raise ValueError(f"{storage_handler} is not supported. Please create StorageHandler for the service.")
        elif storage_handler is not None:
            # this case we store logs on app folder of dropbox, using cloud_storage_handlder
            self.__cloud_handler = storage_handler
            if self.__cloud_handler.refresh_token is None:
                self.__cloud_handler.authenticate()
            self.__use_cloud_storage = True
            if base_path is None:
                self.base_path = "./"
            else:
                self.base_path = base_path
        else:
            self.__cloud_handler = None
            if base_path is None:
                self.base_path = "./"
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

        drive.mount(self.MOUNT_PATH)

    def __store_files_to_cloud_storage(self, file_path):
        try:
            self.__cloud_handler.upload_training_results(self.model_name, [file_path])
            print("file uploaded to cloud storage ")
        except Exception as e:
            print(f"failed to save logs to cloud storage: {e}")

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

    def __append_log(self, log_entry: list, retry_count=0):
        try:
            with open(self.log_file_path, "a") as log_file:
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
                self.__append_log(log_entry, retry_count + 1)
            else:
                self.__cache.append(log_entry)

    def save_params(self, params: dict, model_name, model_version):
        data_folder = os.path.dirname(self.log_file_path)
        param_file_path = os.path.join(data_folder, f"{model_name}_v{model_version}_params.json")
        if "device" in params:
            device = params["device"]
            if not isinstance(device, str):
                params["device"] = str(device)
        with open(param_file_path, mode="w") as fp:
            json.dump(params, fp)
        if self.__use_cloud_storage:
            self.__store_files_to_cloud_storage(param_file_path)

    def save_model(self, model, model_name=None, model_version=None):
        if model is not None:
            data_folder = os.path.dirname(self.log_file_path)
            param_file_path = os.path.join(data_folder, f"{model_name}_v{model_version}.torch")
            torch.save(model.state_dict(), param_file_path)
            if self.__use_cloud_storage:
                self.__store_files_to_cloud_storage(param_file_path)

    def save_checkpoint(self, model, optimizer, scheduler, model_name, model_version, best_loss, **kwargs):
        if model is not None:
            data_folder = os.path.dirname(self.log_file_path)
            model_path = os.path.join(data_folder, f"{model_name}_v{model_version}.torch")
            save_checkpoint(model_path, model, optimizer, scheduler, best_loss, **kwargs)
            if self.__use_cloud_storage:
                self.__store_files_to_cloud_storage(model_path)

    def load_model_checkpoint(
        self,
        create_model_func,
        model_name,
        model_version,
        optimizer_class,
        scheduler_class,
        train=True,
        storage_handler=None,
        optimizer_kwargs={"lr": 1e-3},
        scheduler_kwargs={"step_size": 1, "gamma": 0.95},
        model_folder=None,
    ):
        if model_folder is None:
            data_folder = os.path.dirname(self.log_file_path)
        else:
            data_folder = model_folder
        return load_model_checkpoint(
            create_model_func,
            model_name,
            model_version,
            data_folder,
            optimizer_class,
            scheduler_class,
            train,
            storage_handler,
            optimizer_kwargs,
            scheduler_kwargs,
        )

    def save_logs(self):
        if len(self.__cache) > 0:
            with open(self.log_file_path, "a") as log_file:
                if len(self.__cache) > 0:
                    writer = csv.writer(log_file)
                    writer.writerows(self.__cache)
        if self.__use_cloud_storage:
            self.__store_files_to_cloud_storage(self.log_file_path)

    def add_training_log(self, training_loss, validation_loss, log_entry: list = None):
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
            if self.__cloud_handler is not None:
                file_name = os.path.dirname(self.log_file_path)
                destination_path = f"/{self.model_name}/{file_name}"
                response = self.__cloud_handler.download_file(destination_path, self.log_file_path)
                if response is not None:
                    logs = pd.read_csv(self.log_file_path)
        else:
            try:
                logs = pd.read_csv(self.log_file_path)
            except pd.errors.EmptyDataError:
                logs = None

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
