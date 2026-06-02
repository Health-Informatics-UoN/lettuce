module.exports = {
    "branches": ["main"],
    "tagFormat": "${version}",
    "preset": "angular",
    "repositoryUrl": "https://github.com/Health-Informatics-UoN/<YOUR_REPO_NAME>.git",
    plugins: [
      '@semantic-release/commit-analyzer',
      '@semantic-release/release-notes-generator',
      '@semantic-release/exec',
      '@semantic-release/github',
    ],
    "initialVersion": "0.1.0"
};
