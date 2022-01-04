import tensorflow as tf

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

def apply_gradient(optimizer, model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_val = loss_object(y, logits)

    gradients = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradient(zip(gradients, model.trainable_weights))
    return logits, loss_val

def train_step():
    losses = []
    for step(x_batch_train, y_batch_train) in enumerate(train_data):
        logits, loss_val = apply_gradient(optimizer, model, x_batch_train, y_batch_train)
        losses.append(loss_val)
        train_acc_metrix(y_batch_train, logits)
        return losses

def validation_step():
    losses = []
    for x_val, y_val in val_data:
        val_logits = model(x_val)
        val_loss = loss_object(y_val, val_logits)
        losses.append(val_loss)
        val_acc_metric(y_val, val_logits)
    return losses

# build the model
# this part is actually dummy, there is no model building function defined
# in this script.
model = base_model()

# create training loop
epochs = 10
epoch_train_losses, epoch_val_losses = [], []
for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1}')
    # perform one training step
    # this will calculate and return the training loss
    # and also update the training accuracy metric
    # but we have to call result() of train accuracy metric object
    losses_train = train_step()
    train_acc = train_acc_metric.result()
    
    # perform one validation step
    # this will calculate and return validation loss
    # and also update the validation accuracy metric
    # but we have to call result() for validation accuracy metric object
    losses_val = validation_step()
    val_acc = val_acc_metric()
    
    # we get the mean training loss for each input 
    losses_train_mean = np.mean(losses_train)

    # we get the mean validation loss for each input
    losses_val_mean = np.mean(losses_acc)

    # we append the mean training loss for this step to a list
    epochs_train_losses.append(losses_train_mean)

    # we append the mean validation loss for this step to a list
    epochs_val_losses.append(losses_train_mean)
    
    # last but not least we reset the accuracy metric object state
    # for the next run
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

