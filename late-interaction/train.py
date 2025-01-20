from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="msmarco")):

        config = ColBERTConfig(
            bsize=32,
            root="/path/to/experiments",
        )
        trainer = Trainer(
            triples="/path/to/MSMARCO/triples.train.small.tsv",
            queries="/path/to/MSMARCO/queries.train.small.tsv",
            collection="/path/to/MSMARCO/collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")