#!bin/bash

for m in mol_*.sdf; do
        b=`basename $m .sdf`
        echo Processing ligand $b
        mkdir -p $b
        obabel -i sdf $m -o mol2 -O ${b}.mol2 -h --gen3d
        /home/catkin/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l ${b}.mol2 -o ${b}.pdbqt
        /home/catkin/autodock_vina_1_1_2_linux_x86/bin/vina --config config.txt --ligand ${b}.pdbqt --out ${b}/${b}_out.pdbqt --log log_file/${b}.log
done   