#!/bin/bash

n=1
bsz=10

for size in large medium small xl;do
    for note in data2text webnlg;do
        if [ $n -eq 1 ];then
            let n+=1
            continue
        fi
        echo -e "task"$n"-------------------------------------------------------------------"
        if [ $size == xl -a $note == webnlg ];then
            bsz=5
        else
            bsz=10
        fi
        echo "size: " $size " note: " $note " batch_size: " $bsz
        START=$(date +%s);
        nohup python train_e2e.py --bsz $bsz --mode $note --notes $size &> ../log/$size/$size-$note.log
        END=$(date +%s);
        echo $((END-START)) | awk '{print int($1/3600)"h " int(($1%3600)/60) "min "int($1%60)"s"}'
        if [ $note == data2text ];then
            cd ../text-generation/e2e_results_conv2
            mv data2text* ./$size/
            cd ../../gpt2
        elif [ $note == webnlg ];then
            cd ../text-generation/webNLG_results2
            mv webnlg* ./$size/
            cd ../../gpt2
        else
            echo "some thing wrong"
        fi
        let n+=1
        echo -e "************************************************************************\n\n"
    done
done
