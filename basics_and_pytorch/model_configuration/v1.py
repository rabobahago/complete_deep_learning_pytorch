def make_train_step_fn(model, loss_fn, optimizer):
    def perform_train_step_fn(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return perform_train_step_fn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Linear(1, 1).to(device)
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss_fn = nn.MSELoss(reduction='mean')
performer_train_step = make_train_step_fn(model, loss_fn, optimizer)
performer_train_step
