import torch
import click
from torch import nn
from model import mymodel
from data import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    pass

@click.command()
@click.option("--lr", default=0.003, help="Learning rate used for training")
@click.option("--batch_size", default=128, help="Batch size used for training")
@click.option("--num_epochs", default=10, help="Number of epochs used to train our model")
def train(lr, batch_size, num_epochs):
    print("Training the model")
    print(lr)
    print(batch_size)


    model = mymodel.to(device)
    train_set, _ = mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")

    torch.save(model, "model.pt")

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating the model")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False
    )

    model.eval()
    test_preds = [ ]
    test_labels = [ ]
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print((test_preds == test_labels).float().mean())

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
