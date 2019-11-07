DATASET="L2R_good_with_depth"
GT="./eval/gt"
EN="./eval/encoded"
SA="./eval/encoded_sate"
mkdir -p ${GT}
mkdir -p ${EN}
mkdir -p ${SA}
for FILE in $(ls ./results/${DATASET}/val_sync/images/*_ground_truth.png)
do
	BN="$(basename -- $FILE)"
	cp ${FILE} ${GT}"/"${BN//_ground_truth/}
done
for FILE in $(ls ./results/${DATASET}/val_sync/images/*_encoded.png)
do
	BN="$(basename -- $FILE)"
	cp ${FILE} ${EN}"/"${BN//_encoded/}
done
for FILE in $(ls ./results/${DATASET}/val_sync/images/*_encoded_satellite.png)
do
	BN="$(basename -- $FILE)"
	cp ${FILE} ${SA}"/"${BN//_encoded_satellite/}
done
