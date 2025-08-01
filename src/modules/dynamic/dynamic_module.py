import json
from pathlib import Path

from nebula import Nebula, PEDynamicFeatureExtractor

# from src.utils.interfaces.module import Module
from nebula.models import TransformerEncoderChunks
import nebula
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.metrics import roc_auc_score


model_config = {
    "vocab_size": 50000,
    "maxlen": 512,
    "chunk_size": 64,  # input splitting to chunks
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1,  # binary classification
    "classifier_head": [64],  # classifier ffnn dims
    "layerNorm": False,
    "dropout": 0.3,  #### originale = 0.3
    "norm_first": True,
}

speakeasy_config = (
    Path(__file__).parent.parent / "models/V2/nebula" / "speakeasy_config.json"
)


class DynamicModule:
    def __init__(
        self,
        model_name: str = "nebula",
        fetch_pretrained: bool = False,
        model_path: str = None,
        vocab_path: str = None,
        bpe_model_path: str = None,
    ):
        super().__init__()
        self.model = None
        self.normalizer = None
        self.preprocessor = None

        if fetch_pretrained:
            # if pretrained_path is None:
            #     raise ValueError(
            #         "pretrained_path must be specified when fetch_pretrained is True"
            #     )
            self.load_pretrained_model(
                model_name=model_name,
                vocab_path=vocab_path,
                bpe_model_path=bpe_model_path,
                model_path=model_path,
            )
        else:
            self.build_model(model_name)

    def build_model(self, model_name: str):
        vocab_size = 50000

        preprocessor = nebula.preprocessing.tokenization.JSONTokenizerBPE(
            vocab_size=vocab_size, seq_len=512
        )

        self.normalizer = nebula.preprocessing.pe.PEDynamicFeatureExtractor()

        self.preprocessor = preprocessor

        # setupping model
        self.model = TransformerEncoderChunks(**model_config)

    def load_pretrained_model(
        self,
        vocab_path,
        bpe_model_path,
        model_path,
        default=False,
        model_name: str = "nebula",
    ):
        if model_name == "nebula" and default:
            self.model = Nebula(
                vocab_size=50000,
                seq_len=512,  # pre-trained only for 512
                tokenizer="bpe",  # supports: ["bpe", "whitespace"],
            )
        elif model_name == "nebula":
            self.model = TransformerEncoderChunks(**model_config)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.preprocessor = nebula.preprocessing.tokenization.JSONTokenizerBPE(
                vocab_size=50000,
                seq_len=512,
                vocab=vocab_path,
                model_path=bpe_model_path,
            )
            self.normalizer = nebula.preprocessing.pe.PEDynamicFeatureExtractor()

    def train_module(
        self,
        X=None,
        y=None,
        device=None,
        data_path=None,
        batch_size=64,
        epochs=60,
        validation_split=0.1,
        save_model_path=None,
    ):
        if data_path is not None:
            data = torch.load(data_path)
            X_train = data["X_train"]
            y_train = data["y_train"]
        else:
            X_train, y_train = X, y

        if validation_split is not None:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42
            )

            y_np = y_train_split.numpy()
            num_positive = np.sum(y_np == 1)
            num_negative = np.sum(y_np == 0)
            if num_positive == 0:
                pos_weight = torch.tensor(1.0, dtype=torch.float)
            else:
                pos_weight = torch.tensor(
                    num_negative / num_positive, dtype=torch.float
                )

            # Create TensorDatasets and DataLoaders
            train_dataset = TensorDataset(X_train_split, y_train_split)
            val_dataset = TensorDataset(X_val_split, y_val_split)

            dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            raise NotImplementedError("diocan, dammi tempo")

        # Train the model
        # Extract penultimate part of save_model_path for log_path
        log_path = None
        if save_model_path is not None:
            log_path = os.path.basename(os.path.dirname(save_model_path))

        self.train_model(
            dataloader,
            val_dataloader,
            device=device,
            epochs=epochs,
            pos_weights=pos_weight,
            logger=False,
            log_path=log_path,
            save_model_path=save_model_path,
        )

        # if save_model_path is not None:
        #     print("Saving model...")
        #     os.makedirs(save_model_path, exist_ok=True)
        #     torch.save(
        #         self.model.state_dict(),
        #         save_model_path + "dynamic_model.pt",
        #     )

    def train_model(
        self,
        dataloader,
        val_dataloader,
        device,
        epochs,
        pos_weights,
        logger=False,
        log_path=None,
        save_model_path=None,
    ):
        print("Training model...")

        device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-2
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.3, patience=3
        # )

        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # Create a TensorDataset and DataLoader
        self.model.train()
        self.model.to(device)
        if logger:
            writer = SummaryWriter(
                log_dir="./logs3/" + log_path
            )  # Default log_dir is ./runs

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            self.model.train()
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            ):
                inputs, labels = batch
                labels = labels.unsqueeze(1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(dataloader)
            train_acc = correct / total if total > 0 else 0.0
            if logger:
                writer.add_scalar("Loss/train", epoch_loss, epoch)

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_labels_all = []
            val_outputs_all = []
            with torch.no_grad():
                for val_batch in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"
                ):
                    val_inputs, val_labels = val_batch
                    val_labels = val_labels.unsqueeze(1)
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = self.model(val_inputs)
                    val_outputs_all.extend(torch.sigmoid(val_outputs).cpu().numpy())
                    val_labels_all.extend(val_labels.cpu().numpy())
                    loss = loss_fn(val_outputs, val_labels.float())
                    val_loss += loss.item()

                    val_preds = torch.sigmoid(val_outputs) > 0.5
                    val_correct += (val_preds == val_labels).sum().item()
                    val_total += val_labels.size(0)
                    # Compute AUC if possible

            val_auc = roc_auc_score(val_labels_all, val_outputs_all)

            # Save the model if this is the best AUC so far
            if epoch == 0:
                best_auc = val_auc
                best_model_state = self.model.state_dict()
            else:
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = self.model.state_dict()
                    # torch.save(
                    #     best_model_state,
                    #     os.path.join(
                    #         save_model_path,
                    #         f"dynamic_model.pt",
                    #     ),
                    # )
            val_loss /= len(val_dataloader)
            # scheduler.step(val_loss)
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            if logger:
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Acc/val", val_acc, epoch)
                writer.add_scalar("AUC/val", val_auc, epoch)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        if logger:
            writer.close()

    def preprocess_dataset(self, X, y, save_path=None):
        filtered_data = []

        for sample in X:
            filtered_data.append(self.normalizer.filter_and_normalize_report(sample))

        print("Training tokenizer...")
        self.preprocessor.train(jsonData=filtered_data)

        print("Encoding data...")
        tokenized_data = self.preprocessor.encode(filtered_data)

        X_train = torch.tensor(tokenized_data, dtype=torch.long)
        y_train = torch.tensor(y, dtype=torch.long)

        if save_path is not None:
            print("Moving files...")

            vocab_path = "/data/aponte/repos/obelisk/bpe_vocab.json"
            bpe_model_path = "/data/aponte/repos/obelisk/bpe.model"

            os.makedirs(save_path, exist_ok=True)
            shutil.move(
                vocab_path, os.path.join(save_path, os.path.basename(vocab_path))
            )
            shutil.move(
                bpe_model_path,
                os.path.join(save_path, os.path.basename(bpe_model_path)),
            )

            print("Saving dataset...")
            torch.save(
                {"X_train": X_train, "y_train": y_train},
                os.path.join(save_path, "trainset.pt"),
            )

        return X_train, y_train

    def predict(self, x, device="cpu"):
        self.model.eval()
        device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(device)
        if (isinstance(x, str) or isinstance(x, Path)) and not isinstance(x, dict):
            x = self.normalizer.emulate(str(x))
            if x == None:
                return -1
        elif not isinstance(x, dict):
            raise ValueError(
                "Input must be a string, Path, or a dictionary with 'report' key."
            )
        filtered_x = self.normalizer.filter_and_normalize_report(x)
        tokenized_data = self.preprocessor.encode(filtered_x)
        tokenized_data = torch.tensor(tokenized_data).long().to(device)
        with torch.no_grad():
            logits = self.model(tokenized_data)
            prob = torch.sigmoid(logits)
        return prob.item()

    def emulate(self, x):
        x = self.normalizer.emulate(str(x))
        return x

    def preprocess_test(self, X, y):
        filtered_data = []

        for sample in X:
            filtered_data.append(self.normalizer.filter_and_normalize_report(sample))

        preprocessor = nebula.preprocessing.tokenization.JSONTokenizerBPE(
            vocab_size=50000, seq_len=512
        )

        print("Training tokenizer...")
        preprocessor.train(jsonData=filtered_data)

        print("Encoding data...")
        tokenized_data = self.preprocessor.encode(filtered_data)

        X_train = torch.tensor(tokenized_data, dtype=torch.long)
        y_train = torch.tensor(y, dtype=torch.long)

        # if save_path is not None:
        #     print("Moving files...")

        #     vocab_path = "/data/aponte/repos/obelisk/bpe_vocab.json"
        #     bpe_model_path = "/data/aponte/repos/obelisk/bpe.model"

        #     os.makedirs(save_path, exist_ok=True)
        #     shutil.move(
        #         vocab_path, os.path.join(save_path, os.path.basename(vocab_path))
        #     )
        #     shutil.move(
        #         bpe_model_path,
        #         os.path.join(save_path, os.path.basename(bpe_model_path)),
        #     )

        #     print("Saving dataset...")
        #     torch.save(
        #         {"X_train": X_train, "y_train": y_train},
        #         os.path.join(save_path, "trainset.pt"),
        #     )

        return X_train, y_train
