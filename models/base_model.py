import tensorflow as tf
from keras.src.applications.efficientnet import EfficientNetB4


class ModelBuilder:
    """
    Builder fleksibel untuk berbagai arsitektur backbone.
    Bisa dipakai untuk ensemble atau gonta-ganti model.

    buat build_backbone nya
    masukkan ke build_model
    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2 = tf.keras.regularizers.l2(1e-5)

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

    def build_efficientnet_student(self, version="B4", pretrained=True):
        """
        Buat backbone EfficientNet student.
        """
        backbone_cls = EfficientNetB4 if version == "B4" else getattr(tf.keras.applications, f"EfficientNet{version}")

        backbone = backbone_cls(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=self.input_shape
        )

        # Freeze backbone
        backbone.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        return backbone

    def build_classifier_head(self):
        """
        Head student: global pooling + dense layers + dropout.
        """
        return [
             tf.keras.layers.GlobalAveragePooling2D(),

             tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.l2),
             tf.keras.layers.Dropout(0.5),

             tf.keras.layers.Dense(self.num_classes, activation='softmax', name='student_final_output')
        ]
    def build_model(self, backbone, name="model"):
        model = tf.keras.Sequential(
            [backbone] + self.build_classifier_head(),
            name=name
        )
        return model
