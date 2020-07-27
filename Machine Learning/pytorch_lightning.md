# Pytorch Lightning

## Overview

对于一个 lightning module

```python
class LitModel(LightningModule):

    def __init__(self):
        '''
        	init you model here
        '''
        super().__init__()
        pass

    def forward(self, x):
        """
        	Forward, just like in nn.Module
        """
        pass

    def training_step(self, batch, batch_idx):
        """
        	each training step
        """
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def training_epoch_end(self, outputs):
        """
        	each epoch ends
        """
        pass

    def configure_optimizers(self):
        """
        	return the optimizer for training
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        """
        	Return the training dataloader
        """
        pass
    
    def validation_step(self, batch, batch_idx):
        """
        	each validation step
        """
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """
        	each val epoch end
        """
        tensorboard_logs = {'eval_loss': avg_loss}
         return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        """
        	return a val dataloader
        """
        pass
```

pytorch lightning 实际上是在运行一下步骤

```python
model = LitModel()
torch.set_grad_enabled(True)
model.train()
train_dataloader = model.train_dataloader()
optimizer = model.configure_optimizers()

for epoch in epochs:
    train_outs = []
    for batch in train_dataloader:
        loss = model.training_step(batch)
        loss.backward()
        train_outs.append(loss.detach())

        optimizer.step()
        optimizer.zero_grad()

    if validate_at_some_point:
            torch.set_grad_enabled(False)
            model.eval()
            val_outs = []
            for val_batch in model.val_dataloader:
                val_out = model.validation_step(val_batch)
                val_outs.append(val_out)

            model.validation_epoch_end(val_outs)
            torch.set_grad_enabled(True)
            model.train()

    # optional for logging, etc...
    model.training_epoch_end(train_outs)
```

而且每个 Lightning Module仍然可以当作 nn.Module 使用

```python
model = LitModel()
model.eval()

y_hat = model(x)

model.anything_you_can_do_with_pytorch()
```

## Concept & Ideas

* Research Code

structure of model and how it's trained. This is abstracted by Lighnting Module

* Engineering Code

Training System. eg. Early Stopping, distribution. In Trainer.

* Non-essential code

Inspect Gradients or Log to tensorboard

### Data

`train_dataloader` defines how the batch data is generated.

`prepare_data` relates to stuffs like writing data to disks or so. This is done in the root GPU per node. It's called on the `LOCAL_RANK=0` GPU per node. If your nodes share a same file system, set `Trainer(prepare_data_per_node)=False`, it will be called on node 0, GPU 0 only.

`setup` Use this to build your model if it depends on the dataset you download.

### Logging

When we add the `log` key inside the return dictionary of `training_step` 

```python
def training_step(self, batch, batch_idx):
    # ...
    self.logger.summary.scalar('loss', loss) # acts like tensorboard
```

### Distributed Training

* Delete all the `.to` or `.cuda` calls
* init tensor using `type_as` to cast type of tensor

```python
def forward(self, x):
    z = torch.Tensor(2,3)
    z = z.type_as(x, device = self.device) # using x.device is ok
```

* remove samplers

* make your model pickable

If it's no, the last line of Error will denote where is unpickable.

* select GPU devices

``` python
Trainer(gpus=[1,2]) # select device 1, 2
Trainer(gpus='1, 2') # select device 1, 2
Trainer(gpus=-1) # select all possible devices
```

* TODO

Using `slurm` in a cluster.

### Argument Parser

```python
class LitModel(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parser


# ----------------
# trainer_main.py
# ----------------
from argparse import ArgumentParser
parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--conda_env', type=str, default='some_name')
parser.add_argument('--notification_email', type=str, default='will@email.com')

# add model specific args
parser = LitModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

# init the Trainer
trainer = Trainer.from_argparse_args(args)

# or if you need to pass in callbacks
trainer = Trainer.from_argparse_args(args, checkpoint_callback=..., callbacks=[...])
```

If you want to parse args to multiple models

```python
def main(args):
    dict_args = vars(args)

    # pick model
    if args.model_name == 'gan':
        model = GoodGAN(**dict_args)
    elif args.model_name == 'mnist':
        model = LitMNIST(**dict_args)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--model_name', type=str, default='gan', help='gan or mnist')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == 'gan':
        parser = GoodGAN.add_model_specific_args(parser)
    elif temp_args.model_name == 'mnist':
        parser = LitMNIST.add_model_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)
```