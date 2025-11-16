from datetime import datetime
import tensorflow as tf
from models.finetune import Finetuner


class ModelTrainer:
    def __init__(self, model, loss_fn, hyperparams):
        """
        hyperparams REQUIRED:
        {
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "epochs": 20,
            "patience": 5,
            "model_name": "efficientnet_b3",

            # Fine-tune params:
            "ft_epochs": 20,
            "ft_lr": 1e-5,
            "ft_unfreeze_blocks": 10,
            "ft_weight_decay": 1e-5,
        }
        """
        self.model = model
        self.loss_fn = loss_fn
        self.hp = hyperparams


    # ----------------------------------------------------------
    #  Compile model (Stage 1)
    # ----------------------------------------------------------
    def compile_model(self):
        optimizer_cls = getattr(tf.keras.optimizers, self.hp["optimizer"])
        optimizer = optimizer_cls(learning_rate=self.hp["learning_rate"])

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss_fn,
            metrics=["accuracy"]
        )


    # ----------------------------------------------------------
    #  Stage 1 Training
    # ----------------------------------------------------------
    def train_stage1(self, train_ds, val_ds=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.hp["patience"],
                restore_best_weights=True,
                min_delta=1e-3
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.hp['model_name']}_{timestamp}_best.h5",
                monitor="val_accuracy",
                save_best_only=True
            ),
        ]

        history = self.model.fit(
            train_ds,
            epochs=self.hp["epochs"],
            validation_data=val_ds,
            callbacks=callbacks
        )

        print("✅ Stage 1 training complete.")
        return history


    # ----------------------------------------------------------
    #  Stage 2 Fine-tuning
    # ----------------------------------------------------------
    def train_stage2(self, train_ds, val_ds, backbone):
        print("\n=== FINE-TUNING ===")

        # Unfreeze blocks from hyperparams
        Finetuner.safe_unfreeze_blocks(
            backbone,
            n_blocks=self.hp["ft_unfreeze_blocks"]
        )

        # Cosine LR schedule
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        total_steps = int(steps_per_epoch * self.hp["ft_epochs"])

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.hp["ft_lr"],
            decay_steps=max(total_steps, 1)
        )

        # Try AdamW
        try:
            import tensorflow_addons as tfa
            optimizer = tfa.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=self.hp["ft_weight_decay"],
                clipnorm=1.0
            )
            print("Optimizer: AdamW")
        except:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                clipnorm=1.0
            )
            print("Optimizer: Adam (fallback)")

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss_fn,
            metrics=["accuracy"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                min_delta=1e-2
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.hp['model_name']}_finetuned.h5",
                monitor="val_loss",
                save_best_only=True
            )
        ]

        history = self.model.fit(
            train_ds,
            epochs=self.hp["ft_epochs"],
            validation_data=val_ds,
            callbacks=callbacks
        )

        print("🔥 Fine-tuning complete.")
        return history
