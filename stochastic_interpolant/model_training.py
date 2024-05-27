from stochastic_interpolant.neural_network import NeuralNetwork
import jax
import optax
from jax import grad, vmap
import jax.numpy as jnp
from tqdm.auto import tqdm
import equinox as eqx

def evaluate_test_loss(model, testloader,loss_fun,num_batches = 100):
    avg_loss = 0
    for i,(t,x,y,z) in zip(range(num_batches),testloader):
        avg_loss += loss_fun(model,t,x,y,z)
    return avg_loss / num_batches

def train_model(
  model: NeuralNetwork,
  optim,
  steps,
  train_loader,
  testloader_factory,
  loss_fun,
  print_every,
  num_testloader_batches = 100,
)  -> NeuralNetwork:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model,
        opt_state,
        t,
        x,
        y,
        z
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_fun)(model,t,x,y,z)
        updates, opt_state = optim.update(grads,opt_state,model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    train_losses = jnp.zeros(steps)

    num_test_rounds = 0
    test_losses = jnp.zeros(steps//print_every + 2)
    for step, (t,x,y,z) in zip(tqdm(range(steps)), train_loader):
        model, opt_state, train_loss = make_step(model, opt_state, t,x,y,z)
        train_losses = train_losses.at[step].set(train_loss)
        if jnp.isnan(train_loss):
            print("Nan Obtained")
        if (step % print_every) == 0 or (step == steps - 1):
            testloader = testloader_factory()
            test_loss = evaluate_test_loss(model, testloader,loss_fun,num_testloader_batches)
            test_losses = test_losses.at[num_test_rounds].set(test_loss)
            num_test_rounds += 1
            print("step=" + str(step) + " | train_loss=" + str(train_loss) + " | test_loss= "+ str(test_loss))

    return model,train_losses