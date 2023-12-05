#!/bin/bash

EXPERIMENT_SUFFIX=""
GOLD_CHATGPT_LEARNING_RATE=5e-6
GOLD_LEARNING_RATE=5e-6
NOCHATGPT_GOLD_LEARNING_RATE=5e-6
BATCH_SIZE_PER_GPU=12
GRADIENT_ACCUMULATION_STEPS=20
CHATGPT_GRADIENT_ACCUMULATION_STEPS=20
GOLD_NUM_EPOCHS=20
ONLYCHATGPT_NUM_EPOCHS=20
TEMP=0.3
TEMP1=0.2
TEMP10=0.8
# assuming 8 gpus by default
CUDA_DEVICES=$CUDA_VISIBLE_DEVICES #"0,1,2,3,4,5,6,7"
NUM_GPUS=$(echo $CUDA_DEVICES | awk -F "," '{print NF}') #8
# for 4 gpus
if [ $NUM_GPUS == 4 ]; then
    echo "setting to 4-gpu hyper-parameters"
    GRADIENT_ACCUMULATION_STEPS=40
    CHATGPT_GRADIENT_ACCUMULATION_STEPS=40
fi

EXP_PROCESS="nothing"
while getopts "n:e:s" option; do
    case $option in
        n)  # Experiment name
            EXP_NAME=$OPTARG ;;
        e)  # which process to run
            EXP_PROCESS=$OPTARG ;;
        s)  # custom suffix (default empty)
            EXPERIMENT_SUFFIX=$OPTARG ;;
        \?) # Invalid option
            echo "Error: Invalid option ${option}"
            exit;;
    esac
done



echo "number of gpus:"$NUM_GPUS
echo "CUDA_DEVICES:"$CUDA_DEVICES
echo "process to run: "$EXP_PROCESS

MBPP_TEST_START_TASK_ID=0
MBPP_TEST_END_TASK_ID=601 # (should be exclusive)

PARENT_DIR="models"
EXPERIMENT_DIR="${PARENT_DIR}/${EXP_NAME}"
mkdir -p ${EXPERIMENT_DIR}
echo ${EXPERIMENT_DIR}
CHECKPOINTS_DIR="$(pwd)/checkpoints"


########################################################################################################################################
## Generate the programs for Train + Val dataset using the CodeGen model 

if [ "$EXP_PROCESS" == "gen_student_attempt" ]; then
    echo "generating student attempt"
    for (( i=8 ; i<16 ; i++ )) ; do
        gpu=$(( i - 8 ))
        echo 'Running generate_code_for_alpaca process # ' ${i}
        tmux new-session -d -s m${i} \
        " CUDA_VISIBLE_DEVICES=${gpu} python code/generate_code_for_alpaca.py \
            --num-samples=1 \
            --output-dir=${EXPERIMENT_DIR} \
            --arch=codegen-6B \
            --temperature=${TEMP} --batch-size=1 --debug --proc-id=${i} --num-procs=16 ;"   
    done 
    cat ${EXPERIMENT_DIR}/samples_test_codegen-6B_1shot_temp0.3__* > ${EXPERIMENT_DIR}/samples_test_codegen-6B_1shot_temp0.3.jsonl
fi


########################################################################################################################################
## Evaluate the model on the Code-ALPACA-MBPP data 

if [ "$EXP_PROCESS" == "eval_student_attempt" ]; then
    echo "evaluating student's attempt"
    python eval_alpaca.py \
        --input-file=${EXPERIMENT_DIR}/samples_test_codegen-6B_1shot_temp${TEMP}.jsonl \
        --prompt-column-name prompt --completion-column-name completion --k=1    
    
fi

########################################################################################################################################

## Get ChatGPT feedbacks and rectifications for the above generated programs with --get-openai-feedback. 
## To mention the type of feedback used, use option --feedback-type=auto or --feedback-type=nl 
## Can be incorporated in the previous step itself 

if [ "$EXP_PROCESS" == "get_personalized_refinement" ]; then
    echo "getting personalized refinement from ChatGPT"
    for (( i=0 ; i< 16 ; i++ )) ; do
        echo 'Running generate_code_for_alpaca process # ' ${i}
        tmux new-session -d -s m${i} \
        " python code/generate_code_for_alpaca.py \
            --num-samples=1 \
            --output-dir=${EXPERIMENT_DIR} \
            --arch=codegen-6B \
            --temperature=${TEMP} \
            --debug  \
            --get-openai-feedback --feedback-type=auto --proc-id=${i} --num-procs=16 ;"
    done 

    cat ${EXPERIMENT_DIR}/chatgpt_feedback_test_codegen-16B_1shot_temp0.3__* > ${EXPERIMENT_DIR}/chatgpt_feedback_test_codegen-16B_1shot_temp0.3.jsonl
    
fi



########################################################################################################################################  

## Names of finetuning data files

NOCG_EXPERIMENT_SUFFIX='nochatgpt'
NOFILTER_EXPERIMENT_SUFFIX='nofilter'

GOLD_TRAINING_NOCG_FILE=${EXPERIMENT_DIR}/finetuning_data_gold_${NOCG_EXPERIMENT_SUFFIX}.jsonl
GOLD_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_gold_${EXPERIMENT_SUFFIX}.jsonl
CHATGPT_REFINEMENT_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_chatgpt_rectification_${EXPERIMENT_SUFFIX}.jsonl
GOLD_REFINEMENT_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_gold_rectification_${NOCG_EXPERIMENT_SUFFIX}.jsonl 
CHATGPT_REFINEMENT_TRAINING_NOFILTER_FILE=${EXPERIMENT_DIR}/finetuning_data_chatgpt_rectification_${NOFILTER_EXPERIMENT_SUFFIX}.jsonl

########################################################################################################################################

## Create the finetuning data

if [ "$EXP_PROCESS" == "process_finetune_data" ]; then
    echo "preparing finetuning data"
    python code/create_alpaca_finetuning_data_from_chatgpt_refinements.py \
        --refinement-file=${EXPERIMENT_DIR}/chatgpt_feedback_test_codegen-6B_1shot_temp${TEMP}.jsonl \
        --output-dir=${EXPERIMENT_DIR} --feedback-type=auto \
        --output-gold-file=${GOLD_TRAINING_FILE} \
        --output-gold-nocg-file=${GOLD_TRAINING_NOCG_FILE} \
        --output-refinement-file=${CHATGPT_REFINEMENT_TRAINING_FILE} \
        --output-refinement-nofilter-file=${CHATGPT_REFINEMENT_TRAINING_NOFILTER_FILE} \
        --output-gold-refinement-file=${GOLD_REFINEMENT_TRAINING_FILE} \
        --output-gold-refinement-nofilter-file=${GOLD_REFINEMENT_TRAINING_NOFILTER_FILE} \
        --output-refinement-nofeedback-file=${CHATGPT_REFINEMENT_TRAINING_NOFEEDBACK_FILE} 

fi

########################################################################################################################################

# Create finetuning data with rectification as new gold (similar to text-code task data)

CHATGPT_NEWGOLD_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_chatgpt_newgold_${EXPERIMENT_SUFFIX}.jsonl
CHATGPT_NEWGOLD_RECTONLY_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_chatgpt_newgold_rectification_only_${EXPERIMENT_SUFFIX}.jsonl

python code/create_refinement_as_new_gold_data.py --gold-file=${GOLD_TRAINING_FILE} \
     --chatgpt-refinement-file=${CHATGPT_REFINEMENT_TRAINING_FILE} \
      --output-file=${CHATGPT_NEWGOLD_TRAINING_FILE} \
      --output-rect-file=${CHATGPT_NEWGOLD_RECTONLY_TRAINING_FILE}

########################################################################################################################################

# Create finetuning data for PersD-variants

PersD_combine_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_chatgpt_only.jsonl 
PersD_refine_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_chatgpt_rectification_.jsonl 
InpD_TRAINING_FILE=${EXPERIMENT_DIR}/finetuning_data_chatgpt_gold.jsonl 
python code/create_chatgpt_only_refinement_data.py --gold-file=${GOLD_TRAINING_FILE} \
    --chatgpt-refinement-file=${CHATGPT_REFINEMENT_TRAINING_FILE} \
     --output-file=${PersD_combine_TRAINING_FILE} --output-gold-file=${InpD_TRAINING_FILE}

########################################################################################################################################

# StanD: Finetune the model on gold alpaca data (initialized from scratch)
GOLD_FINETUNE_DIR=${EXPERIMENT_DIR}/gold_finetune_lr${GOLD_LEARNING_RATE}_ga${GRADIENT_ACCUMULATION_STEPS}_${GOLD_NUM_EPOCHS}epochs
mkdir -p ${GOLD_FINETUNE_DIR}

if [ "$EXP_PROCESS" == "finetune_StanD" ]; then
    echo "finetuning StanD"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} deepspeed --master_port 62000 --num_gpus=${NUM_GPUS} \
     code/finetune.py  \
        --do_train \
        --report_to=tensorboard \
        --model_name_or_path=codegen-6B \
        --save_strategy=no \
        --num_train_epochs=${GOLD_NUM_EPOCHS} \
        --learning_rate=${GOLD_LEARNING_RATE} \
        --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        --logging_steps=5 \
        --output_dir=${GOLD_FINETUNE_DIR} \
        --pad_to_max_length \
        --generation_max_length=2048 \
        --max_seq_length=2048 \
        --max_answer_length=2048 \
        --save_total_limit=2 \
        --deepspeed configs/deepspeed_configs.json \
        --bf16 \
        --prompt_column=finetuning_prompt \
        --completion_column=finetuning_completion \
        --overwrite_output_dir \
        --train_file=${GOLD_TRAINING_FILE} #|| exit
fi


########################################################################################################################################

# PersD-combine: using only gold alpaca + rectified data, for those instances where the gold & rectification was correct (initialized from scratch)

GOLD_CHATGPT_ONLY_FINETUNE_DIR=${EXPERIMENT_DIR}/gold_chatgpt_only_finetune_lr${GOLD_CHATGPT_LEARNING_RATE}_ga${CHATGPT_GRADIENT_ACCUMULATION_STEPS}_${ONLYCHATGPT_NUM_EPOCHS}epochs
mkdir -p ${GOLD_CHATGPT_ONLY_FINETUNE_DIR}

if [ "$EXP_PROCESS" == "finetune_PersD_combine" ]; then
    echo "finetuning PersD-combine"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} deepspeed --master_port 62001 --num_gpus=${NUM_GPUS} \
    code/finetune.py  \
        --do_train \
        --report_to=tensorboard \
        --model_name_or_path=codegen-6B \
        --save_strategy=no \
        --num_train_epochs=${ONLYCHATGPT_NUM_EPOCHS} \
        --learning_rate=${GOLD_CHATGPT_LEARNING_RATE} \
        --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps=${CHATGPT_GRADIENT_ACCUMULATION_STEPS} \
        --logging_steps=5 \
        --output_dir=${GOLD_CHATGPT_ONLY_FINETUNE_DIR} \
        --pad_to_max_length \
        --generation_max_length=2048 \
        --max_seq_length=2048 \
        --max_answer_length=2048 \
        --save_total_limit=2 \
        --deepspeed configs/deepspeed_configs.json \
        --bf16 \
        --prompt_column=finetuning_prompt \
        --completion_column=finetuning_completion \
        --overwrite_output_dir \
        --gradient_checkpointing False \
        --train_file ${PersD_combine_TRAINING_FILE} #|| exit
fi

########################################################################################################################################
# PersD: the refinement data is used as new customized gold (text-to-code), only for those instances where the gold & rectification was correct (initialized from scratch)

CHATGPT_NEWGOLD_RECTONLY_FINETUNE_DIR=${EXPERIMENT_DIR}/gold_chatgpt_newgold_finetune_lr${GOLD_CHATGPT_LEARNING_RATE}_ga${CHATGPT_GRADIENT_ACCUMULATION_STEPS}_${CHATGPT_NEWGOLD_NUM_EPOCHS}epochs
mkdir -p ${CHATGPT_NEWGOLD_RECTONLY_FINETUNE_DIR}
if [ "$EXP_PROCESS" == "finetune_PersD" ]; then
    echo "finetuning PersD"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} deepspeed --master_port 62001 --num_gpus=${NUM_GPUS} \
    code/finetune.py  \
        --do_train \
        --report_to=tensorboard \
        --model_name_or_path=codegen-6B \
        --save_strategy=no \
        --num_train_epochs=${ONLYCHATGPT_NUM_EPOCHS} \
        --learning_rate=${GOLD_CHATGPT_LEARNING_RATE} \
        --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps=${CHATGPT_GRADIENT_ACCUMULATION_STEPS} \
        --logging_steps=5 \
        --output_dir=${CHATGPT_NEWGOLD_RECTONLY_FINETUNE_DIR} \
        --pad_to_max_length \
        --generation_max_length=2048 \
        --max_seq_length=2048 \
        --max_answer_length=2048 \
        --save_total_limit=2 \
        --deepspeed configs/deepspeed_configs.json \
        --bf16 \
        --prompt_column=finetuning_prompt \
        --completion_column=finetuning_completion \
        --overwrite_output_dir \
        --gradient_checkpointing False \
        --train_file ${CHATGPT_NEWGOLD_RECTONLY_TRAINING_FILE} #|| exit
fi


########################################################################################################################################

# PersD-refine: finetuning on rectified data only

CHATGPT_ONLY_FINETUNE_DIR=${EXPERIMENT_DIR}/combined_chatgpt_only_finetune_lr${GOLD_CHATGPT_LEARNING_RATE}_ga${CHATGPT_GRADIENT_ACCUMULATION_STEPS}_${ONLYCHATGPT_NUM_EPOCHS}epochs
mkdir -p ${CHATGPT_ONLY_FINETUNE_DIR}

if [ "$EXP_PROCESS" == "finetune_PersD_refine" ]; then
    echo "finetuning PersD-refine"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} deepspeed --master_port 62001 --num_gpus=${NUM_GPUS} \
    code/finetune.py  \
        --do_train \
        --report_to=tensorboard \
        --model_name_or_path=codegen-6B \
        --save_strategy=no \
        --num_train_epochs=${ONLYCHATGPT_NUM_EPOCHS} \
        --learning_rate=${GOLD_CHATGPT_LEARNING_RATE} \
        --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps=${CHATGPT_GRADIENT_ACCUMULATION_STEPS} \
        --logging_steps=5 \
        --output_dir=${GOLD_CHATGPT_ONLY_FINETUNE_DIR} \
        --pad_to_max_length \
        --generation_max_length=2048 \
        --max_seq_length=2048 \
        --max_answer_length=2048 \
        --save_total_limit=2 \
        --deepspeed configs/deepspeed_configs.json \
        --bf16 \
        --prompt_column=finetuning_prompt \
        --completion_column=finetuning_completion \
        --overwrite_output_dir \
        --gradient_checkpointing False \
        --train_file ${PersD_refine_TRAINING_FILE} #|| exit
        
fi

########################################################################################################################################
# Run Evaluation
wait_for_child_tmux_sessions() {
    for (( i=0 ; i<${NUM_GPUS} ; i++ )) ; do
        session_running=1
        while [ $session_running==1 ]
        do
            tmux has-session -t m${i} 2>/dev/null;
            if [ $? != 0 ]; then
                session_running=0
                echo session m${i} finished
                break
            else
                echo 'waiting 1min'
                sleep 60s
            fi
        done
    done
}



if [ "$EXP_PROCESS" == "evaluate_PersD_refine" ] 
then
    model_path=${CHATGPT_ONLY_FINETUNE_DIR}
elif [ "$EXP_PROCESS" == "evaluate_PersD_combine" ]
then
    model_path=${GOLD_CHATGPT_ONLY_FINETUNE_DIR}
elif [ "$EXP_PROCESS" == "evaluate_PersD" ]
then
    model_path=${CHATGPT_NEWGOLD_RECTONLY_FINETUNE_DIR} 
elif [ "$EXP_PROCESS" == "evaluate_StanD" ]
then
    model_path=${GOLD_FINETUNE_DIR}
else
    exit # not evaluating
fi

# Generate programs on humaneval (20 generation for pass@1)

OUTPUT_DIR_HE20_PASS1=${model_path}/output_test_humaneval_output20_pass1_v2
# echo Generate programs on humaneval 20 generation for pass@1  $CHATGPT_ONLY_OUTPUT_DIR_HE40_PASS1
mkdir -p ${OUTPUT_DIR_HE20_PASS1}


for (( i=0 ; i<${NUM_GPUS} ; i++ )) ; do
    echo 'Running generate_code_for_humaneval process # ' ${i}
    TMUX='' tmux new-session -d -s m${i} \
    " CUDA_VISIBLE_DEVICES=${i} python code/generate_code_for_humaneval.py \
        --num-samples=20 \
        --output-dir=${OUTPUT_DIR_HE20_PASS1} \
        --arch=codegen-6B \
        --temperature=${TEMP1} \
        --debug \
        --model-path=${model_path} --refine --num-refine=2 --feedback-type=auto \
        --proc-id=${i} --num-procs=${NUM_GPUS};"
done 
wait_for_child_tmux_sessions

cat ${OUTPUT_DIR_HE20_PASS1}/samples_test_codegen-6B_20shot_temp${TEMP1}_* > ${OUTPUT_DIR_HE20_PASS1}/samples_test_codegen-6B_20shot_temp${TEMP1}.jsonl 
python code/eval_humaneval.py \
    --input-file=${OUTPUT_DIR_HE20_PASS1}/samples_test_codegen-6B_20shot_temp${TEMP1}.jsonl --k=1


# Generate programs on humaneval (100 generation for pass@k)

OUTPUT_DIR_HE100=${model_path}/output_test_humaneval_output100
echo Generate programs on humaneval 100 generation for pass@k $OUTPUT_DIR_HE100
mkdir -p ${OUTPUT_DIR_HE100}


for (( i=0 ; i<${NUM_GPUS} ; i++ )) ; do
    echo 'Running generate_code_for_humaneval process # ' ${i}
    TMUX='' tmux new-session -d -s m${i} \
    " CUDA_VISIBLE_DEVICES=${i} python code/generate_code_for_humaneval.py \
        --num-samples=100 \
        --output-dir=${OUTPUT_DIR_HE100} \
        --arch=codegen-6B \
        --temperature=${TEMP10} \
        --debug \
        --model-path=${model_path} \
        --refine --num-refine=2 \
        --feedback-type=auto \
        --proc-id=${i} --num-procs=${NUM_GPUS}; sleep 20s"
done 
wait_for_child_tmux_sessions

cat ${OUTPUT_DIR_HE100}/samples_test_codegen-6B_100shot_temp${TEMP10}__* > ${OUTPUT_DIR_HE100}/samples_test_codegen-6B_100shot_temp${TEMP10}.jsonl 
python code/eval_humaneval.py \
    --input-file=${OUTPUT_DIR_HE100}/samples_test_codegen-6B_100shot_temp${TEMP10}.jsonl --k=1,5,10,20

