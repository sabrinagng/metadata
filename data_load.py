import os
import zipfile
import scipy.io
import numpy as np
import io

# ----------- 设置路径 -----------
zip_folder = '/Users/km82/Documents/train/Ninapro/DB3'
output_folder = '/Users/km82/Documents/train/Ninapro/DB3_emg_only'
os.makedirs(output_folder, exist_ok=True)

# ----------- 处理每个 zip 文件 -----------
for zip_file in sorted(os.listdir(zip_folder)):
    if not zip_file.endswith('.zip'):
        continue

    zip_path = os.path.join(zip_folder, zip_file)
    subject_id = zip_file.replace('.zip', '')  # e.g., DB2_s1
    out_path = os.path.join(output_folder, f"{subject_id}_emg.npy")

    print(f"\n🔄 Processing: {zip_file}")
    print(f"📍 Will save to: {out_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            mat_files = [f for f in z.namelist() if f.endswith('.mat') and not '/._' in f]
            if not mat_files:
                print(f"  ⚠️ No valid .mat files found in {zip_file}")
                continue
            mat_name = mat_files[0]

            # 加载 .mat
            data_bytes = z.read(mat_name)
            mat = scipy.io.loadmat(io.BytesIO(data_bytes), struct_as_record=False)

            emg = mat.get('emg')
            if emg is None or not isinstance(emg, np.ndarray):
                print(f"  ❌ 'emg' not found or not a valid array in {mat_name}")
                continue

            # 保存 emg 到 .npy
            np.save(out_path, emg)

            # 加载回来验证一下文件确实写入成功
            test = np.load(out_path)
            print(f"  ✅ Saved successfully | Shape: {test.shape}")

    except Exception as e:
        print(f"  ❌ Error processing {zip_file}: {e}")

# ----------- 最后输出结果 -----------
print("\n📂 Final saved files:")
for f in sorted(os.listdir(output_folder)):
    print(" -", f)

