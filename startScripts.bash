python doExperiments_saad.py ${1} ${2}
chmod 777 *.bash
rm nohup.out
for ((i = 0 ; i < ${2} ; i = i + 1 )) ; do
  nohup ./${1}-part-${i}.bash &
done
  