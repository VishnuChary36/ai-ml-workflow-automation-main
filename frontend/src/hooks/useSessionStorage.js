import { useState, useEffect } from 'react';

export function useSessionStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const item = window.sessionStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      if (value === null || value === undefined) {
        window.sessionStorage.removeItem(key);
      } else {
        window.sessionStorage.setItem(key, JSON.stringify(value));
      }
    } catch (error) {
      console.warn('Error setting sessionStorage', error);
    }
  }, [key, value]);

  return [value, setValue];
}
