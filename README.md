# CFZ-Caching

to install :

go into commandline, go to comfyui folder; custom_nodes folder; clone.

'git clone https://github.com/patientx/CFZ-Caching'

cache your prompts for later use. (works with everything except models that require vae input in clip text, for example qwen-image-edit.)

* nodes :
-------------
save-caching : connect it to clip text node to save it so you can load it next time with the load-caching.
load-caching
print-marker : mark a specific point in the workflow in terminal , set a message, clear screen, calculate time it took between points ... (and disable-enable cudnn)
cudnn : simple way to disable-enable cudnn
cudnn-advanced : advanced cudnn node with many settings
