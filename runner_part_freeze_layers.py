"""
Copy and paste this script into runner.py after the location: 
" inside _get_upstream_modules() function, after the model is loaded
i.e.  model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)
## paste the script here"

to freeze layers
"""
## freeze the top 2 layers layers
        
if self.args.upstream == 'hubert_base':
    print("Layer freezing for hubert")
    count = 0
    for name, param in model.named_parameters():
        count += 1
        if (count<177 and count>2):
            
            param.requires_grad = False
            print(name, param.requires_grad)
        elif 'model.layer_norm' in name:
            param.requires_grad = True
            print(name, param.requires_grad)
        elif 'model.final_proj' in name:
            param.requires_grad = True
            print(name, param.requires_grad)
        else:
            print(name, param.requires_grad)
elif self.args.upstream == 'wavlm_base':
    print("Layer freezing for wavlm")
    count = 0
    for name, param in model.named_parameters():
        count += 1
        if count < 207 and count > 1:
            param.requires_grad = False
            print(name, param.requires_grad)
        elif 'model.layer_norm' in name:
            param.requires_grad = True
            print(name, param.requires_grad)
        elif 'model.final_proj' in name:
            param.requires_grad = True
            print(name, param.requires_grad)
        else:
            print(name, param.requires_grad)

## count the number of trainable parameters, added by amit
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", trainable_params)