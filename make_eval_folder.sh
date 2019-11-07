DATASET='L2R_good_with_depth'
GT='./eval/gt'
EN='./eval/encoded'
SA='./eval/encoded_sate'
mkdir -p ${GT}
mkdir -p ${EN}
mkdir -p ${SA}
cp './results/${DATASET}/val_sync/images/*_ground truth.png' ${GT}
cp './results/${DATASET}/val_sync/images/*_encoded.png' ${EN}
cp './results/${DATASET}/val_sync/images/*_encoded_satellite.png' ${SA}
for FILE in $(ls ${GT})
do
	mv ${FILE} ${FILE//_groundtruth/}
done
for FILE in $(ls ${EN})
do
	mv ${FILE} ${FILE//_encoded/}
done
for FILE in $(ls ${SA})
do
	mv ${FILE} ${FILE//_encoded_satellite/}
done
