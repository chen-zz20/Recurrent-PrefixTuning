edition=$1
if [[ $edition == "pytorch" ]] || [[ $edition == "jittor" ]] ; then
    cp ./new-transformers/"$edition"_transformers/ ./PrefixTuning -r
    rm -rf ./PrefixTuning/transformers
    mv ./PrefixTuning/"$edition"_transformers ./PrefixTuning/transformers
    echo finish!
else
    echo you input the wrong edition!
fi