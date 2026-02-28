%% ===========================================================================
%% ScryNeuro Example: Train a CNN on MNIST Handwritten Digits
%% ===========================================================================
%%
%% This example demonstrates the neuro-symbolic workflow:
%%   Prolog orchestrates the entire ML pipeline — model definition, training,
%%   evaluation, and interactive inference — while PyTorch handles the
%%   actual tensor computation on CPU or GPU.
%%
%% Prerequisites:
%%   - Build: cargo build --release && cp target/release/libscryneuro.so ./
%%   - Python deps: pip install torch torchvision
%%
%% Run:
%%   $ LD_LIBRARY_PATH=. scryer-prolog examples/mnist_cnn.pl

:- op(700, xfx, :=).
:- use_module('../prolog/scryer_py').

%% ---------------------------------------------------------------------------
%% Step 1: Define the CNN model in Python
%% ---------------------------------------------------------------------------
%%
%% Architecture:
%%   Conv2d(1,16,3) -> ReLU -> MaxPool
%%   Conv2d(16,32,3) -> ReLU -> MaxPool
%%   Flatten -> Linear(32*5*5, 128) -> ReLU -> Linear(128, 10)

define_model :-
    py_exec_lines([
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
        "class MnistCNN(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)",
        "        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)",
        "        self.pool  = nn.MaxPool2d(2, 2)",
        "        self.fc1   = nn.Linear(32 * 7 * 7, 128)",
        "        self.fc2   = nn.Linear(128, 10)",
        "",
        "    def forward(self, x):",
        "        x = self.pool(F.relu(self.conv1(x)))",
        "        x = self.pool(F.relu(self.conv2(x)))",
        "        x = x.view(x.size(0), -1)",
        "        x = F.relu(self.fc1(x))",
        "        return self.fc2(x)"
    ]),
    format("[Step 1] CNN model defined.~n", []).

%% ---------------------------------------------------------------------------
%% Step 2: Load MNIST dataset via torchvision
%% ---------------------------------------------------------------------------

load_data :-
    py_exec_lines([
        "from torchvision import datasets, transforms",
        "",
        "transform = transforms.Compose([",
        "    transforms.ToTensor(),",
        "    transforms.Normalize((0.1307,), (0.3081,))",
        "])",
        "",
        "train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)",
        "test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)",
        "",
        "train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,  shuffle=True)",
        "test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=256, shuffle=False)",
        "",
        "train_size = len(train_dataset)",
        "test_size  = len(test_dataset)"
    ]),
    py_eval("train_size", TrH), py_to_int(TrH, TrSize), py_free(TrH),
    py_eval("test_size",  TeH), py_to_int(TeH, TeSize), py_free(TeH),
    format("[Step 2] MNIST loaded: ~d training / ~d test samples.~n", [TrSize, TeSize]).

%% ---------------------------------------------------------------------------
%% Step 3: Instantiate model, optimizer, loss function; detect device
%% ---------------------------------------------------------------------------

setup_training :-
    py_exec_lines([
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "model = MnistCNN().to(device)",
        "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)",
        "criterion = nn.CrossEntropyLoss()"
    ]),
    py_eval("str(device)", DevH),
    py_to_str(DevH, Dev),
    py_free(DevH),
    format("[Step 3] Training on device: ~s~n", [Dev]).

%% ---------------------------------------------------------------------------
%% Step 4: Training — run one epoch in Python, report from Prolog
%% ---------------------------------------------------------------------------

%% Train a single epoch entirely in Python, return (avg_loss, accuracy%)
train_one_epoch(Epoch, Loss, Acc) :-
    py_exec_lines([
        "model.train()",
        "_total_loss = 0.0",
        "_correct = 0",
        "_total = 0",
        "for _batch_x, _batch_y in train_loader:",
        "    _batch_x, _batch_y = _batch_x.to(device), _batch_y.to(device)",
        "    optimizer.zero_grad()",
        "    _out = model(_batch_x)",
        "    _loss = criterion(_out, _batch_y)",
        "    _loss.backward()",
        "    optimizer.step()",
        "    _total_loss += _loss.item() * _batch_x.size(0)",
        "    _correct += (_out.argmax(1) == _batch_y).sum().item()",
        "    _total += _batch_x.size(0)",
        "epoch_loss = _total_loss / _total",
        "epoch_acc  = 100.0 * _correct / _total"
    ]),
    py_eval("epoch_loss", LH), py_to_float(LH, Loss), py_free(LH),
    py_eval("epoch_acc",  AH), py_to_float(AH, Acc),  py_free(AH),
    format("  Epoch ~d — loss: ~4f  train_acc: ~2f%~n", [Epoch, Loss, Acc]).

%% Run N epochs using Prolog recursion
train_epochs(Current, Max) :-
    Current > Max, !.
train_epochs(Current, Max) :-
    train_one_epoch(Current, _Loss, _Acc),
    Next is Current + 1,
    train_epochs(Next, Max).

%% ---------------------------------------------------------------------------
%% Step 5: Evaluate on test set
%% ---------------------------------------------------------------------------

evaluate(TestAcc) :-
    py_exec_lines([
        "model.eval()",
        "_correct = 0",
        "_total = 0",
        "with torch.no_grad():",
        "    for _batch_x, _batch_y in test_loader:",
        "        _batch_x, _batch_y = _batch_x.to(device), _batch_y.to(device)",
        "        _out = model(_batch_x)",
        "        _correct += (_out.argmax(1) == _batch_y).sum().item()",
        "        _total += _batch_x.size(0)",
        "test_acc = 100.0 * _correct / _total"
    ]),
    py_eval("test_acc", AH), py_to_float(AH, TestAcc), py_free(AH).

%% ---------------------------------------------------------------------------
%% Step 6: Interactive single-image inference (neural predicate)
%% ---------------------------------------------------------------------------
%%
%% This is the "neuro-symbolic" payoff: a Prolog predicate backed by a
%% trained neural network. `digit/2` maps an image index to its predicted
%% class, which Prolog can then use in logical reasoning.

digit(TestIndex, PredictedClass) :-
    py_from_int(TestIndex, IdxH),
    %% Use a lambda to pass the index from Prolog into the inference pipeline
    py_eval("(lambda i: int(model(test_dataset[i][0].unsqueeze(0).to(device)).argmax(1).item()))", PredFn),
    py_invoke(PredFn, IdxH, ClassH),
    py_to_int(ClassH, PredictedClass),
    py_free(ClassH),
    py_free(PredFn),
    py_free(IdxH).

%% Ground-truth label for comparison
true_label(TestIndex, Label) :-
    py_from_int(TestIndex, IdxH),
    py_eval("(lambda i: int(test_dataset[i][1]))", LabelFn),
    py_invoke(LabelFn, IdxH, LH),
    py_to_int(LH, Label),
    py_free(LH),
    py_free(LabelFn),
    py_free(IdxH).

%% Neuro-symbolic addition: two images -> sum of predicted digits
neuro_add(Idx1, Idx2, Sum) :-
    digit(Idx1, D1),
    digit(Idx2, D2),
    Sum is D1 + D2.

%% Run inference on a few test samples
demo_inference :-
    py_exec("model.eval()"),
    format("~n[Step 6] Single-image inference (neural predicate):~n", []),
    demo_inference_loop(0, 10).

demo_inference_loop(I, Max) :-
    I >= Max, !.
demo_inference_loop(I, Max) :-
    digit(I, Pred),
    true_label(I, Truth),
    ( Pred =:= Truth ->
        Mark = "✓"
    ;
        Mark = "✗"
    ),
    format("  test[~d]: predicted=~d  truth=~d  ~s~n", [I, Pred, Truth, Mark]),
    Next is I + 1,
    demo_inference_loop(Next, Max).

%% ---------------------------------------------------------------------------
%% Step 7: Save the trained model
%% ---------------------------------------------------------------------------

save_model(Path) :-
    py_from_str(Path, PathH),
    py_eval("(lambda p: torch.save(model.state_dict(), p))", SaveFn),
    py_invoke(SaveFn, PathH, _),
    py_free(SaveFn),
    py_free(PathH),
    format("[Step 7] Model saved to: ~s~n", [Path]).

%% ---------------------------------------------------------------------------
%% Main: orchestrate the full pipeline
%% ---------------------------------------------------------------------------

main :-
    format("========================================~n", []),
    format(" ScryNeuro: CNN MNIST Training Pipeline~n", []),
    format("========================================~n~n", []),

    %% Pipeline
    define_model,
    load_data,
    setup_training,

    format("~n[Step 4] Training (3 epochs):~n", []),
    train_epochs(1, 3),

    format("~n[Step 5] Evaluating on test set...~n", []),
    evaluate(TestAcc),
    format("  Test accuracy: ~2f%~n", [TestAcc]),

    demo_inference,

    %% Neuro-symbolic demo
    format("~n[Bonus] Neuro-symbolic addition:~n", []),
    neuro_add(0, 1, Sum01),
    true_label(0, L0), true_label(1, L1),
    format("  digit(test[0])=~d + digit(test[1])=~d => sum=~d  (truth: ~d+~d=",
           [L0, L1, Sum01, L0, L1]),
    TrueSum is L0 + L1,
    format("~d)~n", [TrueSum]),

    save_model("examples/mnist_cnn.pt"),

    format("~n========================================~n", []),
    format(" Pipeline complete!~n", []),
    format("========================================~n", []).

:- initialization((
    py_init,
    main,
    py_finalize
)).
