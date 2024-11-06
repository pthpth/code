# All the runs I want to do - each does about 20% system memory for server 3


# Check that this can run with Sub-TestScript.sh
bash Sub-TestScript.sh "CombineSocialGameRwd_withPenalty"
status=$?
if ! (exit $status); then
    echo "Quick test with no wandb failed"
    exit 1 # exit with error
else
    echo "Quick test with no wandb ran successfully"
fi

echo "This should not print if the above test failed"
# only run this if the above test passed

echo "Doing first set of 4 runs, game reward No_Train, combo reward No_Train, combo penalty Train, random uniform combo"

# bash Sub-GameRewardOnly.sh & bash Sub-ComboReward_NoTrainSI.sh & bash Sub-ComboReward_TrainSI.sh & bash Sub-RandomUniformReward.sh

bash Sub-RewardScriptNoTrainSI.sh "OnlyGameRwd" & bash Sub-RewardScriptNoTrainSI.sh "CombineSocialGameRwd_withPenalty" & bash Sub-RewardScriptTrainSI.sh "CombineSocialGameRwd_withPenalty" & bash Sub-RewardScriptNoTrainSI.sh "RandomRewardsUniform"

wait # wait for them to finish before moving on because memory is limited

echo "Finished first set of 4 runs"

echo "Doing second set of 4 runs, random gaussian, mimic random, pure random, pure penalty"

# bash Sub-RandomGaussianReward.sh & bash Sub-MimicRandomReward.sh & bash Sub-PureRandomReward.sh & bash Sub-PurePenaltyReward.sh

bash Sub-RewardScriptNoTrainSI.sh "RandomRewardGaussian" & bash Sub-RewardScriptNoTrainSI.sh "RandomRewardMimic" & bash Sub-RewardScriptNoTrainSI.sh "OnlyRandomRewardsUniform" & bash Sub-RewardScriptNoTrainSI.sh "OnlySocialRwd_withPenalty"

