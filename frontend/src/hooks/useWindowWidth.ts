import { useEffect, useState } from 'react';

/** `window.innerWidth`, kept current across resizes. */
export function useWindowWidth(): number {
  const [width, setWidth] = useState(() =>
    typeof window === 'undefined' ? 1600 : window.innerWidth,
  );
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', onResize);
    // The width can have changed between the initial state and this effect.
    onResize();
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return width;
}
