import re
import tensorflow as tf
from tabulate import tabulate


class Finetuner:
    """
    Class unfreeze khusus block dengan aman.
    """

    @staticmethod
    def safe_unfreeze_blocks(backbone, n_blocks=2):
        block_dict = {}
        block_pattern = re.compile(r"(block\d+[a-z]*)")

        # ---- kumpulkan block
        for layer in backbone.layers:
            match = block_pattern.search(layer.name)
            block_name = match.group(1) if match else "other"

            if block_name not in block_dict:
                block_dict[block_name] = []
            block_dict[block_name].append(layer)

        sorted_blocks = sorted(
            [b for b in block_dict.keys() if b != "other"],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )

        blocks_to_unfreeze = sorted_blocks[-n_blocks:]
        print("\n🔓 Unfreezing blocks:", blocks_to_unfreeze)

        for block_name in blocks_to_unfreeze:
            for layer in block_dict[block_name]:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

        for block_name in sorted_blocks:
            if block_name not in blocks_to_unfreeze:
                for layer in block_dict[block_name]:
                    layer.trainable = False

        # ---- tampilkan tabel
        table = []
        idx = 0
        for block_name in sorted_blocks:
            for layer in block_dict[block_name]:
                table.append([
                    idx,
                    layer.name,
                    block_name,
                    layer.trainable
                ])
                idx += 1

        print("\n========== BLOCK TRAINABLE TABLE ==========")
        print(tabulate(
            table,
            headers=["Index", "Layer", "Block", "Trainable"],
            tablefmt="grid"
        ))

        return blocks_to_unfreeze
