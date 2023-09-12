## Loading custom checkpoint, added by the user
print("Name of the upstream model", self.args.upstream)
print("Model.training status:", model.training)
print("Loading custom pretrained upstream model, added by the user")
if self.args.upstream == 'hubert_base':
    custom_ckpt_path = "path_to_checkpoints/states-3600.ckpt"
elif self.args.upstream == 'wavlm_base':
    custom_ckpt_path = "path_to_checkpoints/states-3600.ckpt"
else:
    print("No custom pretrained model found")
    sys.exit()
tuned_ckpt = torch.load(custom_ckpt_path)
print("The path for the custom model is:", custom_ckpt_path)
model.load_state_dict(tuned_ckpt['Upstream']) # Loading the custom upstream model
print("Model.training status after loading weights from custom model:", model.training)
