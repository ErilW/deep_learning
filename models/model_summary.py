from tabulate import tabulate


class ModelSummary:
    @staticmethod
    def show(model):
        rows = []
        for i, layer in enumerate(model.layers):
            rows.append([
                i,
                layer.name,
                layer.__class__.__name__,
                getattr(layer, "output_shape", "-"),
                layer.count_params(),
                layer.trainable
            ])

        print("\n========== FULL MODEL SUMMARY ==========")
        print(tabulate(
            rows,
            headers=["Index", "Layer", "Type", "Output", "Params", "Trainable"],
            tablefmt="grid"
        ))

