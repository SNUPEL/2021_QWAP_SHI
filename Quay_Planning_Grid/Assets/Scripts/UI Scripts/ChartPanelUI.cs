using System.Collections;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

[RequireComponent(typeof(RectTransform))]
public class ChartPanelUI : MonoBehaviour,
    IPointerDownHandler, IBeginDragHandler, IDragHandler, IEndDragHandler,
    IPointerEnterHandler, IPointerExitHandler, IPointerClickHandler
{
    [Header("Refs")]
    public Canvas canvas;                   // 이 Panel이 속한 Canvas (Inspector에서 할당)
    public CanvasGroup canvasGroup;         // 선택(없으면 자동 추가)
    public Button CloseButton;              // 다시 제자리로 돌아가게 하는 버튼
    public Shadow shadow;                   // 선택(없으면 효과 생략)

    [Header("Animation Settings")]
    public float animationTime = 0.3f;
    private Vector3 expandedScale = Vector3.one * 3f;


    [Header("Hover Settings")]
    private float hoverScale = 1.03f;      // 드래그/호버 시 살짝 확대
    public float hoverAlpha = 1f;      // 드래그 시 살짝 투명
    public bool bringToFrontOnDrag = true;

    [Header("Bounds")]
    public bool clampToParent = true;     // 부모 Rect 영역 안으로 제한

    private RectTransform rt;
    private RectTransform parentRt;

    private Vector3 originalPos;
    private Vector3 currentScale = Vector3.one;
    private Vector3 originalScale = Vector3.one;
    private bool isExpanded = false;
    private Coroutine animCoroutine;


    //private Vector2 pointerOffset;        // 클릭 지점과 패널 pivot 간 상대 위치
    Vector2 grabOffset;
    private float originalAlpha = 1f;

    // 서브캔버스 정렬을 위한 옵션(필요 시 on)
    public bool useOwnSubCanvas = true;   // 패널에 별도 Canvas를 달아 정렬
    Canvas selfCanvas;                    // overrideSorting, sortingOrder 조절
    static int s_OrderSeed = 100;         // 포커싱될수록 증가

    void Awake()
    {
        rt = GetComponent<RectTransform>();
        if (canvas == null) canvas = GetComponentInParent<Canvas>();
        parentRt = rt.parent as RectTransform;

        originalPos = rt.anchoredPosition;
        currentScale = rt.localScale;

        if (CloseButton != null)
            CloseButton.onClick.AddListener(RestorePanel);

        if (canvasGroup == null)
        {
            canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null) canvasGroup = gameObject.AddComponent<CanvasGroup>();
        }
        currentScale = rt.localScale;
        originalAlpha = canvasGroup.alpha;

        if (useOwnSubCanvas)
        {
            selfCanvas = GetComponent<Canvas>();
            if (selfCanvas == null) selfCanvas = gameObject.AddComponent<Canvas>();
            selfCanvas.overrideSorting = true;  // 자체 정렬 사용
            selfCanvas.sortingOrder = ++s_OrderSeed;
            // GraphicRaycaster가 없다면 추가
            if (GetComponent<GraphicRaycaster>() == null) gameObject.AddComponent<GraphicRaycaster>();
        }
    }

    private void ExpandPanel()
    {
        if (animCoroutine != null) StopCoroutine(animCoroutine);
        animCoroutine = StartCoroutine(AnimatePanel(
            rt.anchoredPosition, Vector2.zero,
            rt.localScale, expandedScale,
            animationTime
            ));
        currentScale = expandedScale;
        isExpanded = true;
        CloseButton.gameObject.SetActive(true);     
        rt.SetAsLastSibling();                      // 항상 최상단에 배치
    }

    public void RestorePanel()
    {
        if (!isExpanded) return;
        if (animCoroutine != null) StopCoroutine(animCoroutine);
        animCoroutine = StartCoroutine(AnimatePanel(
            rt.anchoredPosition, originalPos,
            rt.localScale, originalScale,
            animationTime ));
        currentScale = originalScale;
        isExpanded = false;
        CloseButton.gameObject.SetActive(false);
    }

    private IEnumerator AnimatePanel(Vector2 fromPos, Vector2 toPos, Vector3 fromScale, Vector3 toScale, float time)
    {
        float elapsed = 0f;
        while (elapsed < time)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / time);

            rt.anchoredPosition = Vector2.Lerp(fromPos, toPos, t);
            rt.localScale = Vector3.Lerp(fromScale, toScale, t);

            yield return null;
        }
        rt.anchoredPosition = toPos;
        rt.localScale = toScale;
    }
    public void OnPointerClick(PointerEventData eventData)
    {
        if (isExpanded) return;
        ExpandPanel();
    }


    // 마우스가 패널 위에 들어왔을 때(호버) – 시각효과(선택)
    public void OnPointerEnter(PointerEventData eventData)
    {
        BringToFront();
        rt.localScale = Vector3.Lerp(rt.localScale, currentScale * hoverScale, 1f);
        if (shadow != null) shadow.enabled = true;
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        // 드래그 중이 아니면 원상복구
        if (!eventData.dragging)
        {
            rt.localScale = currentScale;
            if (shadow != null) shadow.enabled = false;
        }
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        BringToFront();
        // 부모 좌표계 기준의 마우스 위치
        RectTransformUtility.ScreenPointToLocalPointInRectangle(
            parentRt, eventData.position, eventData.pressEventCamera, out var mouseLocalInParent);

        // 현재 패널 위치(anchoredPosition)와의 오프셋 저장
        grabOffset = rt.anchoredPosition - mouseLocalInParent;
    }

    public void OnBeginDrag(PointerEventData eventData)
    {
        canvasGroup.blocksRaycasts = false;
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (parentRt == null) return;

        // 드래그 중에도 항상 부모 좌표계로 변환
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(
            parentRt, eventData.position, eventData.pressEventCamera, out var mouseLocalInParent))
        {
            Vector2 newPos = mouseLocalInParent + grabOffset; // 동일 좌표계에서 합산
            rt.anchoredPosition = newPos; // 필요하면 여기서 클램프
        }
    }

    public void OnEndDrag(PointerEventData eventData)
    {
        canvasGroup.blocksRaycasts = true;
    }
    void BringToFront()
    {
        // 동일 부모 내에서는 최상단 배치
        rt.SetAsLastSibling();

        // 서로 다른 서브캔버스/패널끼리도 확실히 위로 오게
        if (useOwnSubCanvas && selfCanvas != null)
        {
            selfCanvas.overrideSorting = true;
            selfCanvas.sortingOrder = ++s_OrderSeed; // 클릭할 때마다 최신 order 부여
        }
    }

    private Vector2 ClampToParent(Vector2 desiredAnchoredPos)
    {
        // 패널/부모의 사각형 계산
        var panel = rt.rect;
        var parent = parentRt.rect;

        // 패널 pivot을 고려한 경계
        float left = parent.xMin + panel.width * rt.pivot.x;
        float right = parent.xMax - panel.width * (1f - rt.pivot.x);
        float bottom = parent.yMin + panel.height * rt.pivot.y;
        float top = parent.yMax - panel.height * (1f - rt.pivot.y);

        float x = Mathf.Clamp(desiredAnchoredPos.x, left, right);
        float y = Mathf.Clamp(desiredAnchoredPos.y, bottom, top);
        return new Vector2(x, y);
    }
}
