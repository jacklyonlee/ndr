import trainer


def run():
    for bsize in [64, 128, 256, 512]:
        for hsize in [64, 128]:
            lp, knn, _ = trainer.train(
                "AE", z_dim=256, hidden_dim=hsize, batch_size=bsize, n_epochs=8
            )
            print(f"B{bsize}, H{hsize}: LP={lp}, KNN={knn}")


if __name__ == "__main__":
    run()
