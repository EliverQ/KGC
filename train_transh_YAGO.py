

import my_oke

from my_oke.module.model import TransH
from my_oke.module.loss_fun import MarginLoss
from my_oke.module.SamplingStrategy import NegativeSampling

from my_oke.config import Trainer, Tester

from my_oke.dataloader import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/YAGO3-10/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/YAGO3-10/", "link")

# define the model
transh = TransH(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transh, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
trainer.run()
transh.save_checkpoint('./checkpoint/transh.ckpt')

# test the model
transh.load_checkpoint('./checkpoint/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)