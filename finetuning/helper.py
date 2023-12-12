def load_training_log(log_txt_path : str, print_res : bool = False) -> dict:
    log_data = {}
    current_epoch = None
    current_phase = None

    # Read the file
    with open(log_txt_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Check for epoch line
            if line.startswith('Epoch'):
                parts = line.split()
                current_epoch = int(parts[1])
                current_phase = parts[2].strip('[]:')

                # Initialize dictionary for the epoch and phase
                if current_epoch not in log_data:
                    log_data[current_epoch] = {}
                if current_phase not in log_data[current_epoch]:
                    log_data[current_epoch][current_phase] = {}

            # Check for loss and accuracy lines
            elif line.startswith('train_') or line.startswith('validation_'):
                metric, value = line.split(':')
                value = float(value.strip())
                log_data[current_epoch][current_phase][metric] = value

    if print_res:
        for epoch, data in log_data.items():
            print(f"Epoch {epoch}:")
            for phase, metrics in data.items():
                print(f"  {phase}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value}")

    return log_data

def read_epoch(
        log_data : dict,
        split : str, # Validation/Training
        epoch : int
    ) -> dict:
    remap = {}
    t = log_data[epoch][split]
    a = "train" if split == "Training" else "validation"
    remap[a + '_loss'] = t[a+'_loss']
    remap[a + '_average_accuracy'] = t[a + '_average_accuracy']
    remap[a + 'accuracy forest'] = t[a + '_accuracy_class_0']
    remap[a + 'accuracy shrubland'] = t[a + '_accuracy_class_1']
    remap[a + 'accuracy grassland'] = t[a + '_accuracy_class_2']
    remap[a + 'accuracy wetlands'] = t[a + '_accuracy_class_3']
    remap[a + 'accuracy croplands'] = t[a + '_accuracy_class_4']
    remap[a + 'accuracy urban'] = t[a + '_accuracy_class_5']
    remap[a + 'accuracy barren'] = t[a + '_accuracy_class_6']
    remap[a + 'accuracy water'] = t[a + '_accuracy_class_7']
    remap[a + 'accuracy cacao'] = t[a + '_accuracy_class_8']
    return remap    