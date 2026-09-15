import assert from 'assert';
import { VolumetricPreprocessor } from '../src/volumetric_preprocessor.js';
import { MultiViewClassifier } from '../src/multi_view_classifier.js';
import { KLGradingEngine } from '../src/kl_grading_engine.js';

console.log('🧪 Starting RSNA Knee AI Automated Verification Suite (Kaggle RSNA 2026)...\n');

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
  sagittalVolume: Array.from({ length: 32 }, () => [0.85, 0.90, 0.78]),
  coronalVolume: Array.from({ length: 32 }, () => [0.72, 0.65, 0.70]),
  axialVolume: Array.from({ length: 32 }, () => [0.20, 0.25, 0.30])
};

const result = MultiViewClassifier.predict(mockStudy);
assert(typeof result.predictions.aclTear === 'number', 'Must predict ACL tear probability');
assert(typeof result.predictions.meniscusTear === 'number', 'Must predict Meniscus tear probability');
assert(typeof result.predictions.cartilageAbnormality === 'number', 'Must predict Cartilage abnormality probability');
console.log(`   📊 Diagnostic Probabilities for Study [${result.studyId}]:`);
console.log(`      • ACL Tear: ${(result.predictions.aclTear * 100).toFixed(1)}% | Meniscus Tear: ${(result.predictions.meniscusTear * 100).toFixed(1)}%`);

// Test 4: Kellgren-Lawrence (KL) Ordinal Classification
console.log('4️⃣ Testing Kellgren-Lawrence (KL) 5-Grade Ordinal Classification...');
// Case A: Grade 0 (Normal)
const normalCase = KLGradingEngine.classifyKLGrade({ jointSpaceWidthMm: 4.8, osteophytesPresent: false });
assert(normalCase.predictedGrade === 0, 'Normal case must be Grade 0');

// Case B: Grade 3 (Moderate OA)
const moderateCase = KLGradingEngine.classifyKLGrade({
  jointSpaceWidthMm: 2.1,
  osteophytesPresent: true,
  subchondralSclerosis: true
});
assert(moderateCase.predictedGrade === 3, 'Moderate case must be Grade 3');
assert(moderateCase.gradCamHeatmap.medialCompartmentAttention > 0.8, 'Grad-CAM attention must be elevated on medial compartment');
console.log(`   ✅ KL Grade 0 (Normal): "${normalCase.interpretation}"`);
console.log(`   ✅ KL Grade 3 (Moderate): "${moderateCase.interpretation}" (Grad-CAM Medial Attention: ${moderateCase.gradCamHeatmap.medialCompartmentAttention})`);

// Test 5: Quadratic Weighted Kappa (QWK) Metric Verification
console.log('5️⃣ Testing Competition Primary Metric: Quadratic Weighted Kappa (QWK)...');
const actualLabels = [0, 1, 2, 3, 4, 2, 3, 1, 0, 4];
const predictedLabels = [0, 1, 2, 3, 4, 2, 2, 1, 0, 4]; // 9/10 match, one off by 1
const qwk = KLGradingEngine.calculateQWK(actualLabels, predictedLabels);
assert(qwk >= 0.90, `Expected QWK >= 0.90, got ${qwk}`);
console.log(`   🏆 Benchmark Quadratic Weighted Kappa (QWK): ${qwk} (Excellent inter-rater agreement)`);

console.log('\n🎉 ALL 5 RSNA KNEE AI & KL GRADING TESTS PASSED WITH 100% SUCCESS!\n');
