ChargeNet_nChannels_22_May_2020-11h05
    minimizer&activation: adam&relu
    trained on set: train
    special notes:
ChargeNet_flat_24_Sep_2020-07h48
    minimizer&activation: adam&relu
    trained on set: 149999 flat
    special notes:

DOMNet_26_Jun_2020-16h49
    minimizer&activation: adam&relu
    trained on set: valid
    special notes: trained with memory mapping
DOMNet_reduced_22_Jul_2020-15h18
    minimizer&activation: adam&relu
    trained on set: train
    special notes: uses just DOMs from allowed_doms.npy, new data_generator with container size 20
DOMNet_reduced_ranger_23_Jul_2020-21h16
    minimizer&activation: ranger&mish
    trained on set: train
    special notes: uses just DOMs from allowed_doms.npy, new data_generator with container size 20
DOMNet_reduced_ranger_25_Jul_2020-21h45
    minimizer&activation: ranger&mish
    trained on set: train
    special notes: uses just DOMs from allowed_doms.npy, new data_generator with container size 20, bigger NN
DOMNet_reduced_ranger_28_Jul_2020-19h18
    minimizer&activation: ranger&mish
    trained on set: train
    special notes: uses just DOMs from allowed_doms.npy, new data_generator with container size 20, BatchNorm after input, ResNet structure
DOMNet_reduced_new_10_Sep_2020-06h51
    minimizer&activation: adam&mish+x^2
    trained on set: train
    special notes: uses just DOMs from allowed_doms.npy, new data_generator with container size 20, E weights, ln(r) as input

Hitnet_13_Jul_2020-14h18
    minimizer&activation: adam&relu
    trained on set: train
    special notes: 
Hitnet_ranger_14_Jul_2020-08h03
    minimizer&activation: ranger&mish
    trained on set: train
    special notes:
HitNet_ranger_30_Jul_2020-15h49
    minimizer&activation: ranger&mish
    trained on set: train
    special notes: new shuffleing (just "in DOM")
HitNet_ranger_total_flat_23_Sep_2020-17h47
    minimizer&activation: ranger&mish
    trained on set: 149999 flat
    special notes:
    
LayerNet_05_Jun_2020-10h18
    minimizer&activation: adam&relu
    trained on set: train
    special notes: 

StringNet_05_Jun_2020-13h32
    minimizer&activation: adam&relu
    trained on set: train
    special notes: 