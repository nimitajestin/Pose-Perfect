import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    if (error.code === 'ECONNREFUSED') {
      throw new Error('Server is not running. Please start the API server.');
    }
    if (error.response?.status === 404) {
      throw new Error('API endpoint not found');
    }
    if (error.response?.status === 500) {
      throw new Error('Internal server error');
    }
    throw error;
  }
);

export const poseAPI = {
  // Get list of available poses
  async getPoses() {
    try {
      const response = await api.get('/poses');
      const data = response.data;

      // Only the pose-specific models are playable. The legacy poses in
      // `data.legacy_poses` are the sample keypoint files that setup.py writes to
      // target_poses/ - /score/base64 rejects them with a 404, and they have no
      // reference image, so listing them puts unplayable poses in the game.
      const specificPoses = data.pose_specific_models || [];

      return specificPoses.map(pose => ({
        name: pose,
        type: 'pose-specific'
      }));
    } catch (error) {
      console.error('Error fetching poses:', error);
      throw new Error(error.message || 'Failed to fetch poses');
    }
  },

  // Get detailed model information
  async getPoseModels() {
    try {
      const response = await api.get('/poses/models');
      return response.data;
    } catch (error) {
      console.error('Error fetching pose models:', error);
      throw new Error(error.message || 'Failed to fetch pose models');
    }
  },

  // Score a pose from base64 image
  async scorePose(imageBase64, pose) {
    try {
      // Extract pose name from object or use string directly
      const poseName = typeof pose === 'object' ? pose.name : pose;

      const payload = {
        image_base64: imageBase64, // Keep the full data URL for the new endpoint
        pose_name: poseName,
        conf_threshold: 0.5
      };

      const response = await api.post('/score/base64', payload);
      return response.data;
    } catch (error) {
      console.error('Error scoring pose:', error);
      throw new Error(error.message || 'Failed to score pose');
    }
  },

  // Batch score multiple poses
  async scorePoseBatch(images) {
    try {
      const payload = {
        images: images.map(img => ({
          image_base64: img.imageBase64.replace(/^data:image\/[a-z]+;base64,/, ''),
          pose_name: typeof img.pose === 'object' ? img.pose.name : img.pose,
          player_id: img.playerId
        }))
      };

      const response = await api.post('/score/batch', payload);
      return response.data;
    } catch (error) {
      console.error('Error batch scoring poses:', error);
      throw new Error(error.message || 'Failed to batch score poses');
    }
  },

  // Get model info for a specific pose
  async getModelInfo(poseName) {
    try {
      const response = await api.get(`/models/${poseName}/info`);
      return response.data;
    } catch (error) {
      console.error('Error fetching model info:', error);
      throw new Error(error.message || 'Failed to fetch model info');
    }
  },

  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('API server is not responding');
    }
  }
};

// Utility functions
export const utils = {
  // Convert file to base64
  fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  },

  // Validate base64 image
  isValidBase64Image(base64String) {
    if (!base64String) return false;

    const base64Pattern = /^data:image\/(jpeg|jpg|png|gif|webp);base64,/;
    return base64Pattern.test(base64String);
  },

  // Format API errors for display
  formatApiError(error) {
    if (error.response?.data?.detail) {
      return error.response.data.detail;
    }
    if (error.message) {
      return error.message;
    }
    return 'An unexpected error occurred';
  },

  // Reference images that actually exist in Frontend/public/poses
  poseImageFiles: [
    'ArcherPose.jpg',
    'Dab.jpg',
    'FighterPose.jpg',
    'FlexPose.jpg',
    'KarateKidPose.jpg',
    'SRKPose.jpg',
    'StatueOfLibertyPose.jpg',
    'SurprisedPose.jpg',
    'ThumkebaazPose.jpg',
    'TPose.jpeg',
    'TreePose.jpg',
    'WarriorPose.jpg'
  ],

  // Poses that have no image of their own but can borrow another one
  poseImageAliases: {
    chuggayegurupose: 'WarriorPose.jpg',
    chudgayeguru: 'WarriorPose.jpg'
  },

  // Get the public URL of a pose's reference image, or null when there is none.
  // The API mixes naming styles - pose-specific models are CamelCase ("WarriorPose")
  // while the legacy sample poses created by setup.py are snake_case ("warrior_pose",
  // "t_pose") - so match on a normalised key instead of the exact string.
  getPoseImagePath(pose) {
    const poseName = typeof pose === 'object' ? pose?.name : pose;
    if (!poseName) return null;

    const normalize = str => String(str).toLowerCase().replace(/[^a-z0-9]/g, '');

    const byName = {};
    for (const file of utils.poseImageFiles) {
      byName[normalize(file.replace(/\.[^.]+$/, ''))] = file;
    }

    const key = normalize(poseName);
    // Also try the "Pose" suffix, so "warrior" finds WarriorPose.jpg.
    const file = byName[key] || byName[`${key}pose`] || utils.poseImageAliases[key];

    return file ? `/poses/${file}` : null;
  },

  // Format pose name for display (convert camelCase to readable format)
  formatPoseName(poseName) {
    return poseName
      .replace(/([A-Z])/g, ' $1') // Add space before uppercase letters
      .replace(/^./, str => str.toUpperCase()) // Capitalize first letter
      .trim(); // Remove any leading/trailing spaces
  }
};

export default api;