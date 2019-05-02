# Find all non-script data files.
FILENAMES=$(find . -name "*.pyc" -o -name "*.dat" -o -name "*.ldat" -o -name "*.mod" -o -name "*.lmod" -o -name "*.png" -o -name "*.svg")
BASENAMES=$(echo $FILENAMES | xargs basename | xargs echo)

# Look for references to given filenames in .py and .ipynb files.
rm referenced_files.txt all_files.txt
for pat in $BASENAMES
do
	grep -ohr -F $pat . --include "*.py" --include "*.ipynb" --include "*.mk" >> referenced_files.txt 
	echo $pat >> all_files.txt
done

# Find filenames that aren't referenced in any scripts.
sort -u -o all_files.txt -- all_files.txt
sort -u -o referenced_files.txt -- referenced_files.txt
comm -23 all_files.txt referenced_files.txt > orphaned_files.txt

# Cleanup.
rm all_files.txt referenced_files.txt
