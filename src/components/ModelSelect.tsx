import { useEffect, useState } from 'react';
import { chatAPI, ModelsResponse } from '@/lib/api';

interface Props {
  value: string | undefined;
  onChange: (v: string) => void;
  type: 'generation' | 'embedding';
  className?: string;
  placeholder?: string;
}

export function ModelSelect({ value, onChange, type, className, placeholder }: Props) {
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    chatAPI
      .getModels()
      .then((res: ModelsResponse) => {
        if (!mounted) return;
        const list = type === 'generation' ? res.generation_models : res.embedding_models;
        setModels(list);
        setLoading(false);
      })
      .catch((e) => {
        if (!mounted) return;
        setError(String(e));
        setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [type]);

  if (loading) {
    return (
      <select className={className} disabled>
        <option>Loading…</option>
      </select>
    );
  }
  if (error || models.length === 0) {
    return (
      <select className={className} disabled>
        <option>No models</option>
      </select>
    );
  }

  return (
    <select
      className={`px-2 py-1 bg-gray-800 text-white rounded ${className || ''}`}
      value={value || ''}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="" disabled>
        {placeholder || `Select ${type === 'generation' ? 'LLM' : 'embed model'}`}
      </option>
      {models.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
} 