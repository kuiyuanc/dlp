"""
implement nn with 2 hidden layers
    at least one transformation: linear, CNN, ...
    at least one activation: sigmoid, tanh, ReLU, ...
    perform backpropagation to update weights
log
    print loss per 5k epoch during training
    print prediction & accuracy during testing
visualization
    plot prediction vs ground truth
    plot loss vs epoch
    print accuracy
extra
    different optimizers
    different activation functions
    convolution layers
train & test with same data
"""

from argparse import ArgumentParser

import numpy as np

import activation
from layer import Linear
from loss import BCE
from model import Model
from optimizer import SGD
from util import generate_linear, generate_XOR_easy, show_loss, show_result


def get_model(args, input_size, output_size):
    act = eval(f"activation.{args.activation}")

    model = Model()
    model.add(Linear(input_size, args.hidden))
    model.add(act())
    model.add(Linear(args.hidden, args.hidden))
    model.add(act())
    model.add(Linear(args.hidden, output_size))
    model.add(activation.Sigmoid())
    model.compile()

    return model


def train(X, Y, model, optimizer, input_size):
    losses_per_epoch = []
    for e in range(args.epoch):
        losses = []
        for x, y in zip(X, Y):
            x, y = x.reshape(input_size, -1), y.reshape(1, 1)
            prediction = model(x)
            loss, delta = BCE(prediction, y)
            model.backward(delta, prediction)
            optimizer.step()
            losses.append(loss)
        losses_per_epoch.append(np.mean(losses))

        if (e + 1) % 500 == 0:
            print(f"Epoch {e + 1:>4}/{args.epoch} Loss: {np.mean(losses):.8f}")
    return losses_per_epoch

def validate(X, Y, model, input_size):
    losses = []
    for x, y in zip(X, Y):
        x, y = x.reshape(input_size, -1), y.reshape(1, 1)
        prediction = model(x)
        loss, _ = BCE(prediction, y)
        losses.append(loss)

    x_all, y_all = X.reshape(-1, input_size, 1), Y.reshape(-1, 1, 1)
    prediction = model(x_all)
    prediction = (prediction > 0.5).astype(int)
    accuracy = np.mean(prediction == y_all)

    for i, (yp, y) in enumerate(zip(prediction, y_all)):
        print(f"Iter {i:2d}\t| Ground truth: {y[0][0]} | Predicted: {yp[0][0]}")
    print(f"loss={np.mean(losses):.5f}, accuracy={accuracy * 100:.2f}%")


def main(args):
    X, Y = generate_linear(n=100) if args.problem == "linear" else generate_XOR_easy()
    input_size, output_size = 2, 1

    model = get_model(args, input_size, output_size)

    optimizer = SGD(model, lr=args.lr)

    losses_per_epoch = train(X, Y, model, optimizer, input_size)

    validate(X, Y, model, input_size)

    show_loss(losses_per_epoch, name=args.problem)
    show_result(X, Y, (model(X.reshape(-1, input_size, 1)) > 0.5).astype(int))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--problem", type=str, default="linear", choices=["linear", "XOR"])
    parser.add_argument("--activation", type=str, default="Sigmoid", choices=["Sigmoid", "Identity"])
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD"])
    parser.add_argument("--epoch", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--hidden", type=int, default=16)
    args = parser.parse_args()
    main(args)
