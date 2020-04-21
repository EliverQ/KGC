import my_oke

from my_oke.module.model import TransE
from my_oke.module.loss_fun import SigmoidLoss
from my_oke.module.SamplingStrategy import NegativeSampling

from my_oke.config import Trainer, Tester

from my_oke.dataloader import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./datasets/WN18RR/", 
	batch_size = 2000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./datasets/WN18RR/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 1024, 
	p_norm = 1,
	norm_flag = False,
	margin = 6.0)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = SigmoidLoss(adv_temperature = 1),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 3000, alpha = 2e-5, use_gpu = True, opt_method = "adam")
trainer.run()
transe.save_checkpoint('./checkpoint/transe_2.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe_2.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
