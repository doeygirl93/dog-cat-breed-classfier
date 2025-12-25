import torch
from torch import nn
import data_setup, model_nn_arch, epoch_loop, save_logic
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# get datasetup
train_loader, test_loader, CLASS_NAMES = data_setup.create_data_setup(BATCH_SIZE, NUM_WORKERS)

# get model
model = model_nn_arch.define_nn_arch().to(DEVICE)

#loss n optim
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# START THE TRAINING LOOP

epoch_loop.train(model, train_loader, test_loader, loss_fn, optimizer, epochs=EPOCHS, device=DEVICE)

save_logic.save_model(model, target_dir="models", model_name="cat_dog_breed_classfier.pth")



#pretty much done!!!!


#CHANGES:
#added torch bc issure with training
