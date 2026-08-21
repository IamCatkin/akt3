#!/bin/bash
trap 'echo "Error occurred at line $LINENO"' ERR


preprocess_mol() {
  mol_file_path=$1
  mol_file_dir=$2
  mol_name=$3

  tgt_dir="$mol_file_dir/$mol_name/"
  if [ ! -d "$tgt_dir" ]; then
    mkdir -p "$tgt_dir"
    echo "$tgt_dir has been created"
  fi

  gro_path="$tgt_dir/molecule.gro"
  top_path="$tgt_dir/molecule.top"
  itp_path="$tgt_dir/molecule.itp"

  printf "$mol_file_path\n2\n$gro_path\n1\n2\n4\n$top_path\n$itp_path\n0\n" | ./sobtop || {
    echo "Error in preprocess_mol for $mol_name"
    return 1
  }
  
  echo '#ifdef POSERS' >> "$itp_path"
  echo '#include "posre.itp"' >> "$itp_path"
  echo '#endif' >> "$itp_path"
}

combine() {
  work_path=$1
  mol_name=$2

  mol_gro_path="$work_path/results/$mol_name/molecule.gro"
  protein_gro_path="$work_path/protein/protein.gro"

  n_mol_line=$(wc -l < "$mol_gro_path")  
  n_mol_atom=$(sed -n "2p" "$mol_gro_path")  
  n_protein_atom=$(sed -n "2p" "$protein_gro_path")  

  total_atom=$((n_protein_atom + n_mol_atom)) 
  out_file="$work_path/kotori_results/$mol_name/complex.gro"

  
  head -n 1 "$protein_gro_path" > "$out_file"  
  echo "$total_atom" >> "$out_file"  
  head -n -1 "$protein_gro_path" | tail -n +3 >> "$out_file"  
  head -n -1 "$mol_gro_path" | tail -n +3 >> "$out_file"  
  tail -n 1 "$protein_gro_path" >> "$out_file"  

  src_topol_file="$work_path/protein/topol.top"
  tgt_topol_file="$work_path/kotori_results/$mol_name/topol.top"


  sed '/; Include chain topologies/{n; s|^#include "|#include "../../protein/|; :a; n; /^; Include /!{ s|^#include "|#include "../../protein/|; ba; }}' "$src_topol_file" > "$tgt_topol_file"

  line_number=$(grep -n '#include "topol_Protein_chain_A.itp"' "$src_topol_file" | cut -d ":" -f1)
  sed -i "${line_number}i #include \"molecule.itp\"" "$tgt_topol_file"

  echo "$mol_name     1" >> "$tgt_topol_file"
}

gen_posre_file() {
  work_path=$1
  mol_name=$2

  mol_gro_path="$work_path/kotori_results/$mol_name/molecule.gro"
  out_file="$work_path/kotori_results/$mol_name/posre.itp"
  printf "0\n" | gmx genrestr -f "$mol_gro_path" -o "$out_file" || {
    echo "Error in generating posre for $mol_name"
    return 1
  }
}

run() {
  work_path=$1
  mol_name=$2
  gpu=$3

  mol_root="$work_path/kotori_results/$mol_name"

  box_in="$mol_root/complex.gro"
  box_out="$mol_root/protein_box.gro"

  water_out="$mol_root/protein_sol.gro"
  topol_file="$mol_root/topol.top"

  tpr_out="$mol_root/em.tpr"
  iron_out="$mol_root/system.gro"

  em_gro="$mol_root/em.gro"
  eq_tpr="$mol_root/eq.tpr"
  eq_gro="$mol_root/eq.gro"
  md_tpr="$mol_root/md.tpr"

  {
    gmx editconf -f "$box_in" -o "$box_out" -d 1.0 -bt cubic && \
    gmx solvate -cp "$box_out" -o "$water_out" -p "$topol_file" && \
    gmx grompp -f "$work_path/em.mdp" -c "$water_out" -r "$water_out" -p "$topol_file" -o "$tpr_out" -maxwarn 1 && \
    printf "15\n" | gmx genion -s "$tpr_out" -p "$topol_file" -o "$iron_out" -neutral -conc 0.15 && \
    gmx grompp -f "$work_path/em.mdp" -c "$iron_out" -r "$iron_out" -p "$topol_file" -o "$tpr_out" && \
    pushd "$mol_root" && gmx mdrun -v -deffnm em -ntmpi 1 -ntomp 16 -gpu_id "$gpu" && popd && \
    gmx grompp -f "$work_path/eq.mdp" -c "$em_gro" -p "$topol_file" -o "$eq_tpr" -r "$em_gro" -maxwarn 1 && \
    pushd "$mol_root" && gmx mdrun -v -deffnm eq -ntmpi 1 -ntomp 16 -gpu_id "$gpu" && popd && \
    gmx grompp -f "$work_path/md.mdp" -c "$eq_gro" -r "$eq_gro" -p "$topol_file" -o "$md_tpr" && \
    pushd "$mol_root" && gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 16 -gpu_id "$gpu" && popd && echo 'done'
  } || {
    echo "Error in run step for $mol_name"
    return 1
  }
}

analyse() {
  work_path=$1
  mol_name=$2

  mol_root="$work_path/results/$mol_name"

  md_trr="$mol_root/md.trr"
  md_xtc="$mol_root/md.xtc"
  md_edr="$mol_root/md.edr"
  md_tpr="$mol_root/md.tpr"
  {
    gmx trjconv -f $md_trr -o $md_xtc && \
    printf "15\n" | gmx energy -f $md_edr -o "$mol_root/${mol_name}_energy.xvg" && \

    printf "1\n1\n" | gmx rms -f $md_xtc -s $md_tpr -o "$mol_root/${mol_name}_rmsd_protein.xvg" && \
    printf "4\n4\n" | gmx rms -f $md_xtc -s $md_tpr -o "$mol_root/${mol_name}_rmsd_backbone.xvg" && \
    printf "1\n13\n" | gmx rms -f $md_xtc -s $md_tpr -o "$mol_root/${mol_name}_rmsd_lig.xvg" && \
    printf "13\n13\n" | gmx rms -f $md_xtc -s $md_tpr -o "$mol_root/${mol_name}_rmsd_lig_lig.xvg" && \
    printf "1\n13\n" | gmx hbond -f $md_xtc -s $md_tpr -num "$mol_root/${mol_name}_hbnum.xvg"
  } || {
    echo "Error in run step for $mol_name"
    return 1
  }
}

main() {
  work_path=$1
  mol_file_path=$2
  gpu_id=$3

  mol_file_name=$(basename "$mol_file_path")
  mol_file_dir=$(dirname "$mol_file_path")
  mol_name="${mol_file_name%.*}"

  # preprocess_mol "$mol_file_path" "$mol_file_dir" "$mol_name" && \
    # combine "$work_path" "$mol_name" && \
    # gen_posre_file "$work_path" "$mol_name" && \
    # run "$work_path" "$mol_name" "$gpu_id" && \
    analyse "$work_path" "$mol_name" || {
    echo "Skipping molecule $mol_name due to errors"
    return 1
  }
}

work_path=$1
mol2_dir=$2
gpu_id=$3

find "$work_path/$mol2_dir" -type f | while read -r file; do
    main "$work_path" "$file" "$gpu_id" || continue
done