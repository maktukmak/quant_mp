import torch

from quant_mp.models import LinNet

# TODO: Adapt to new API


def train(model, device, train_loader, optimizer, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    loss_sum = 0.0
    loss_vec = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if isinstance(model, LinNet):
            data = data.flatten(start_dim=1)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        if (batch_idx + 1) % 30 == 0:
            loss_vec.append(loss_sum / 30)
            loss_sum = 0.0
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        # if batch_idx > 0:
        #     s_vec.append(model.fci.qweight.s.item())

    return loss_vec


def test(model, device, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    loss_vec_test = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if isinstance(model, LinNet):
                data = data.flatten(start_dim=1)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    loss_vec_test.append(test_loss)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return loss_vec_test
