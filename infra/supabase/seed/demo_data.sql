-- Demo data for development and testing

-- Insert initial prompt versions
INSERT INTO prompt_versions (version_id, version_name, prompt_text) VALUES
    ('v1', 'v1', 'You are a helpful assistant that answers questions based on the provided context.'),
    ('v2', 'v2', 'You are an expert assistant that provides detailed, well-structured answers based on the provided context.')
ON CONFLICT (version_id) DO NOTHING;

