import os

# paths a Annotated y Cropped
annotated_path = "/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated"
cropped_path = "/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped"

# Leer los nombres de las carpetas en ambos directorios
annotated = set(os.listdir(annotated_path))
cropped = set(os.listdir(cropped_path))

# Calcular diferencias e intersecciones
only_in_annotated = sorted(list(annotated - cropped))
only_in_cropped = sorted(list(cropped - annotated))
in_both = sorted(list(annotated & cropped))
cropped_and_both = sorted(list(only_in_cropped + in_both))

# Guardar
def save_list(filename, data):
    with open(filename, "w") as f:
        f.write("\n".join(data))

save_list("only_in_annotated.txt", only_in_annotated)
save_list("only_in_cropped.txt", only_in_cropped)
save_list("both_in_annotated_and_cropped.txt", in_both)
save_list("cropped_and_both.txt", cropped_and_both)


# Optional: print quick summary
print(f"Only in Annotated: {len(only_in_annotated)} folders")
print(f"Only in Cropped: {len(only_in_cropped)} folders")
print(f"In both: {len(in_both)} folders")
