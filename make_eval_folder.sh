DATASET="L2R_good_with_depth"
GT="./eval/gt"
EN="./eval/encoded"
SA="./eval/encoded_sate"
mkdir -p ${GT}
mkdir -p ${EN}
mkdir -p ${SA}
for FILE in $(ls ./results/${DATASET}/val_sync/images/*_ground*.png)
do
	BN="$(basename -- $FILE)"
	mv ${FILE} ${GT}"/"${BN//_ground\ truth/}
done
for FILE in $(ls ${EN})
do
	BN="$(basename -- $FILE)"
	mv ${FILE} ${EN}"/"${BN//_encoded/}
done
for FILE in $(ls ${SA})
do
	BN="$(basename -- $FILE)"
	mv ${FILE} ${SA}"/"${BN//_encoded_satellite/}
done