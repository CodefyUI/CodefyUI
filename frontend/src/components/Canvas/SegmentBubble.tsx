import { ViewportPortal, useViewport } from '@xyflow/react';
import { useMemo } from 'react';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { computeSegmentNodes } from '../../utils/segmentPath';
import type { Node as FlowNode } from '@xyflow/react';
import type { NodeData } from '../../types';

const BUBBLE_PAD = 28;
const BUBBLE_RADIUS = 28;
const BUBBLE_FILL = 'rgba(255, 180, 80, 0.22)';
const BUBBLE_STROKE = 'rgba(255, 140, 0, 0.6)';
const BUBBLE_STROKE_ACTIVE = 'rgba(255, 149, 0, 0.95)';
const BADGE_FILL = '#ff9500';

interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

function nodeBBox(n: FlowNode<NodeData>): BBox {
  const w = n.measured?.width ?? n.width ?? 200;
  const h = n.measured?.height ?? n.height ?? 80;
  return { x: n.position.x, y: n.position.y, w, h };
}

function unionBBox(boxes: BBox[]): BBox | null {
  if (boxes.length === 0) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const b of boxes) {
    if (b.x < minX) minX = b.x;
    if (b.y < minY) minY = b.y;
    if (b.x + b.w > maxX) maxX = b.x + b.w;
    if (b.y + b.h > maxY) maxY = b.y + b.h;
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

/**
 * Canvas overlay that renders every persistent SegmentGroup as its own
 * light-orange bubble with HEAD / TAIL badges and a per-bubble × close
 * button. Clicking the × removes only that segment. The active segment
 * gets a slightly bolder stroke so you can tell which one the Inspector
 * is currently showing.
 */
export function SegmentBubble() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const segmentGroups = activeTab.segmentGroups;
  const activeSegment = activeTab.activeSegment;
  const nodes = activeTab.nodes;
  const edges = activeTab.edges;
  const removeSegmentGroup = useTabStore((s) => s.removeSegmentGroup);
  const setActiveSegment = useTabStore((s) => s.setActiveSegment);
  const { zoom } = useViewport();
  const { t } = useI18n();

  const renderable = useMemo(() => {
    return segmentGroups
      .map((g) => {
        const set = computeSegmentNodes(g.headNodeId, g.tailNodeId, nodes, edges);
        if (set.size === 0) return null;
        const head = nodes.find((n) => n.id === g.headNodeId);
        const tail = nodes.find((n) => n.id === g.tailNodeId);
        if (!head || !tail) return null;
        const segmentNodes = nodes.filter((n) => set.has(n.id));
        const union = unionBBox(segmentNodes.map(nodeBBox));
        if (!union) return null;
        return { group: g, head, tail, union };
      })
      .filter((v): v is NonNullable<typeof v> => v !== null);
  }, [segmentGroups, nodes, edges]);

  if (renderable.length === 0) return null;

  const stroke = 2 / zoom;
  const strokeActive = 3 / zoom;

  return (
    <ViewportPortal>
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          overflow: 'visible',
          pointerEvents: 'none',
          zIndex: 0,
        }}
      >
        {renderable.map(({ group, head, tail, union }) => {
          const isActive = activeSegment?.id === group.id;
          const rect = {
            x: union.x - BUBBLE_PAD,
            y: union.y - BUBBLE_PAD,
            w: union.w + BUBBLE_PAD * 2,
            h: union.h + BUBBLE_PAD * 2,
          };
          return (
            <g key={group.id}>
              {/* Clicking the rect focuses this segment. pointer-events: stroke
                  means only the border is interactive, so clicks in the middle
                  fall through to the canvas for panning / node selection. */}
              <rect
                x={rect.x}
                y={rect.y}
                width={rect.w}
                height={rect.h}
                rx={BUBBLE_RADIUS}
                ry={BUBBLE_RADIUS}
                fill={BUBBLE_FILL}
                stroke={isActive ? BUBBLE_STROKE_ACTIVE : BUBBLE_STROKE}
                strokeWidth={isActive ? strokeActive : stroke}
                pointerEvents="stroke"
                onClick={() => setActiveSegment(group)}
                style={{ cursor: 'pointer' }}
              />
              <Badge box={nodeBBox(head)} anchor="top-left" text="HEAD" />
              <Badge box={nodeBBox(tail)} anchor="bottom-right" text="TAIL" />
              <CloseButton
                rect={rect}
                zoom={zoom}
                onClick={() => removeSegmentGroup(group.id)}
                title={t('segment.removeThis')}
              />
            </g>
          );
        })}
      </svg>
    </ViewportPortal>
  );
}

interface BadgeProps {
  box: BBox;
  anchor: 'top-left' | 'bottom-right';
  text: string;
}

function Badge({ box, anchor, text }: BadgeProps) {
  const w = text.length * 8 + 14;
  const h = 18;
  const x = anchor === 'top-left' ? box.x - 4 : box.x + box.w - w + 4;
  const y = anchor === 'top-left' ? box.y - h - 4 : box.y + box.h + 4;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} ry={4} fill={BADGE_FILL} />
      <text
        x={x + w / 2}
        y={y + h / 2 + 4}
        textAnchor="middle"
        fill="#ffffff"
        fontSize={11}
        fontWeight={700}
        fontFamily="ui-monospace, SFMono-Regular, monospace"
        style={{ letterSpacing: '0.06em' }}
      >
        {text}
      </text>
    </g>
  );
}

interface CloseButtonProps {
  rect: BBox;
  zoom: number;
  onClick: () => void;
  title: string;
}

/**
 * Small × in the top-right corner of the bubble. Uses pointer-events='all'
 * just on this element so clicks reach the handler even though the parent
 * SVG is pointer-events='none'.
 */
function CloseButton({ rect, onClick, title }: CloseButtonProps) {
  const r = 10;
  const cx = rect.x + rect.w - 6;
  const cy = rect.y + 6;
  return (
    <g
      pointerEvents="all"
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      style={{ cursor: 'pointer' }}
    >
      <title>{title}</title>
      <circle cx={cx} cy={cy} r={r} fill="#1a1a1a" stroke={BADGE_FILL} strokeWidth={1.5} />
      <line
        x1={cx - 4}
        y1={cy - 4}
        x2={cx + 4}
        y2={cy + 4}
        stroke={BADGE_FILL}
        strokeWidth={1.6}
        strokeLinecap="round"
      />
      <line
        x1={cx + 4}
        y1={cy - 4}
        x2={cx - 4}
        y2={cy + 4}
        stroke={BADGE_FILL}
        strokeWidth={1.6}
        strokeLinecap="round"
      />
    </g>
  );
}
