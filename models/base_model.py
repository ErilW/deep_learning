import tensorflow as tf

class ModelBuilder:
    """
    Builder fleksibel untuk berbagai arsitektur backbone.
    Bisa dipakai untuk ensemble atau gonta-ganti model.
    """

    def __init__(self, input_shape, num_classes, l2_reg=1e-5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2 = tf.keras.regularizers.l2(l2_reg)

    def build_efficientnet(self, version="B3", pretrained=True):
        backbone_cls = getattr(tf.keras.applications, f"EfficientNet{version}")

        backbone = backbone_cls(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=self.input_shape,
        )

        # freeze full backbone
        backbone.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        return backbone

    # ======================
    # HEAD SEKARANG FLEXIBLE
    # ======================
    def build_classifier_head(self, layers_config=None):
        """
        layers_config: list of dict, contoh:
        [
            {"units": 1024, "activation": "relu", "dropout": 0.4, "batchnorm": True},
            {"units": 512, "activation": "relu", "dropout": 0.35, "batchnorm": True},
        ]
        """
        head_layers = [tf.keras.layers.GlobalAveragePooling2D()]

        if layers_config is None:
            # default head
            layers_config = [
                {"units": 1024, "activation": "relu", "dropout": 0.4, "batchnorm": True},
                {"units": 512, "activation": "relu", "dropout": 0.35, "batchnorm": True},
                {"units": 256, "activation": "relu", "dropout": 0.3, "batchnorm": True},
            ]

        for layer_cfg in layers_config:
            units = layer_cfg.get("units", 256)
            activation = layer_cfg.get("activation", "relu")
            dropout = layer_cfg.get("dropout", 0.0)
            batchnorm = layer_cfg.get("batchnorm", True)

            head_layers.append(tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=self.l2))
            if batchnorm:
                head_layers.append(tf.keras.layers.BatchNormalization())
            if dropout > 0:
                head_layers.append(tf.keras.layers.Dropout(dropout))

        # output layer
        head_layers.append(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        return head_layers

    def build_model(self, backbone, head_config=None, name="model"):
        """
        build_model sekarang bisa menerima head_config untuk flexible head
        """
        head_layers = self.build_classifier_head(layers_config=head_config)
        model = tf.keras.Sequential([backbone] + head_layers, name=name)
        return model
