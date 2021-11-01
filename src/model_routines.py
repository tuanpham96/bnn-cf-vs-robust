
# Config
#   1. Data config
batch_size = 100
task_sequences = ['MNIST', 'p-MNIST', 'p-MNIST', 'p-MNIST']

#   2. Model config
layer_dims = [(28,28), 512, 500, 300, 10]

#   3. Learning config
meta = 0.5
learning_rate = 0.001


# Data
ds_args = dict(batch_size=batch_size, return_dict=True, return_iters=False)
for full_task in task_sequences:
    task, action = parse_task(full_task)
    ds.append(create_data_iter(task, action, **ds_args))

# Model
net = BNN(layer_dims)

# Optim + loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = Adam_meta(meta = meta, learning_rate = learning_rate)

# Stats
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



for i in range(len(test_loader_list)):
    data['acc_test_tsk_'+str(i+1)], data['loss_test_tsk_'+str(i+1)] = [], []


for task_idx, task in enumerate(train_loader_list):
    if not(args.beaker or args.si):
        optimizer = Adam_meta(model.parameters(), lr = lrs[task_idx], meta = meta, weight_decay = args.decay)

    for epoch in range(1, epochs+1):
        train(model, task, task_idx, optimizer, device, args)
        train_accuracy, train_loss = test(model, task, device, verbose=True)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    # This loop is for BNN parameters having 'org' attribute
    for p in list(model.parameters()): # blocking weights with org value greater than a threshold by setting grad to 0
        if hasattr(p,'org'):
            p.data.copy_(p.org)


    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# for epoch in range(EPOCHS):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     test_loss.reset_states()
#     test_accuracy.reset_states()

for images, labels in train_ds:
    train_step(images, labels)

def train(model, train_loader, current_task_index, optimizer, device, args,
          prev_cons=None, prev_params=None, path_integ=None, criterion = torch.nn.CrossEntropyLoss())

def test(model, test_loader, device, criterion = torch.nn.CrossEntropyLoss(reduction='sum'), verbose = False):

def update_W(model, W, p_old, args):
