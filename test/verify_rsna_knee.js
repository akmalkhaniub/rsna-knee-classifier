import assert from 'assert';
import { VolumetricPreprocessor } from '../src/volumetric_preprocessor.js';
import { MultiViewClassifier } from '../src/multi_view_classifier.js';

console.log('🧪 Starting RSNA Knee Abnormality Detection Automated Verification Suite (Kaggle)...\n');

// Test 1: Slice Depth Standardization
console.log('1️⃣ Testing Volumetric Slice Depth Standardization...');
const raw19Slices = Array.from({ length: 19 }, (_, i) => [[i, i + 1], [i + 2, i + 3]]);
const standardized32 = VolumetricPreprocessor.standardizeDepth(raw19Slices, 32);
assert(standardized32.length === 32, `Expected 32 slices, got ${standardized32.length}`);
console.log(`   ✅ Resampled 19-slice DICOM series into standardized 32-slice volumetric tensor.`);

// Test 2: Percentile Intensity Normalization
console.log('2️⃣ Testing Percentile Intensity Normalization...');
const rawVoxels = [0, 15, 22, 105, 450, 890, 1400, 2048, 3100, 4095];
const normalized = VolumetricPreprocessor.normalizeVoxelIntensity(rawVoxels);
assert(normalized.every(v => v >= 0.0 && v <= 1.0), 'All voxels must be normalized to [0, 1]');
console.log('   ✅ Normalized sample voxel intensities:', normalized.slice(0, 5).join(', '), '...');

// Test 3: Multi-View Classification & Cross-Sequence Fusion
console.log('3️⃣ Testing Multi-View Diagnostic Prediction & Fusion...');
const mockStudy = {
  studyId: 'study_patient_00492',
  sagittalVolume: Array.from({ length: 32 }, () => [0.85, 0.90, 0.78]), // High T2 signal in ACL intercondylar notch
  coronalVolume: Array.from({ length: 32 }, () => [0.72, 0.65, 0.70]),  // Meniscus tear hyperintensity
  axialVolume: Array.from({ length: 32 }, () => [0.20, 0.25, 0.30])     // Intact cartilage
};

const result = MultiViewClassifier.predict(mockStudy);
assert(typeof result.predictions.aclTear === 'number', 'Must predict ACL tear probability');
assert(typeof result.predictions.meniscusTear === 'number', 'Must predict Meniscus tear probability');
assert(typeof result.predictions.cartilageAbnormality === 'number', 'Must predict Cartilage abnormality probability');

console.log(`   📊 Diagnostic Probabilities for Study [${result.studyId}]:`);
console.log(`      • ACL Tear Probability: ${(result.predictions.aclTear * 100).toFixed(1)}% (Positive: ${result.diagnosticImpression.hasACLInjury})`);
console.log(`      • Meniscus Tear Probability: ${(result.predictions.meniscusTear * 100).toFixed(1)}% (Positive: ${result.diagnosticImpression.hasMeniscusInjury})`);
console.log(`      • Cartilage Defect Probability: ${(result.predictions.cartilageAbnormality * 100).toFixed(1)}% (Positive: ${result.diagnosticImpression.hasCartilageInjury})`);

assert(result.diagnosticImpression.hasACLInjury === true, 'Should detect ACL injury from high sagittal hyperintensity');
assert(result.diagnosticImpression.hasMeniscusInjury === true, 'Should detect Meniscus injury from coronal hyperintensity');
assert(result.diagnosticImpression.hasCartilageInjury === false, 'Should correctly classify intact cartilage');

console.log('\n🎉 ALL RSNA KNEE ABNORMALITY DETECTION TESTS PASSED WITH 100% SUCCESS!\n');
