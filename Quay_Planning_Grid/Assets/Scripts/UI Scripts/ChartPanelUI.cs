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
    public Canvas canvas;
    public CanvasGroup canvasGroup;
    public Button CloseButton;
    public Shadow shadow;

    [Header("Animation Settings")]
    public float animationTime = 0.3f;
    private Vector3 expandedScale = Vector3.one * 3f;


    [Header("Hover Settings")]
    private float hoverScale = 1.03f;
    public float hoverAlpha = 1f;
    public bool bringToFrontOnDrag = true;

    [Header("Bounds")]
    public bool clampToParent = true;

    private RectTransform rt;
    private RectTransform parentRt;

    private Vector3 originalPos;
    private Vector3 currentScale = Vector3.one;
    private Vector3 originalScale = Vector3.one;
    private bool isExpanded = false;
    private Coroutine animCoroutine;


    Vector2 grabOffset;
    private float originalAlpha = 1f;

    public bool useOwnSubCanvas = true;
    Canvas selfCanvas;
    static int s_OrderSeed = 100;

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
            selfCanvas.overrideSorting = true;
            selfCanvas.sortingOrder = ++s_OrderSeed;
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
        rt.SetAsLastSibling();
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


    public void OnPointerEnter(PointerEventData eventData)
    {
        BringToFront();
        rt.localScale = Vector3.Lerp(rt.localScale, currentScale * hoverScale, 1f);
        if (shadow != null) shadow.enabled = true;
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        if (!eventData.dragging)
        {
            rt.localScale = currentScale;
            if (shadow != null) shadow.enabled = false;
        }
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        BringToFront();
        RectTransformUtility.ScreenPointToLocalPointInRectangle(
            parentRt, eventData.position, eventData.pressEventCamera, out var mouseLocalInParent);

        grabOffset = rt.anchoredPosition - mouseLocalInParent;
    }

    public void OnBeginDrag(PointerEventData eventData)
    {
        canvasGroup.blocksRaycasts = false;
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (parentRt == null) return;

        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(
            parentRt, eventData.position, eventData.pressEventCamera, out var mouseLocalInParent))
        {
            Vector2 newPos = mouseLocalInParent + grabOffset;
            rt.anchoredPosition = newPos;
        }
    }

    public void OnEndDrag(PointerEventData eventData)
    {
        canvasGroup.blocksRaycasts = true;
    }
    void BringToFront()
    {
        rt.SetAsLastSibling();

        if (useOwnSubCanvas && selfCanvas != null)
        {
            selfCanvas.overrideSorting = true;
            selfCanvas.sortingOrder = ++s_OrderSeed;
        }
    }

    private Vector2 ClampToParent(Vector2 desiredAnchoredPos)
    {
        var panel = rt.rect;
        var parent = parentRt.rect;

        float left = parent.xMin + panel.width * rt.pivot.x;
        float right = parent.xMax - panel.width * (1f - rt.pivot.x);
        float bottom = parent.yMin + panel.height * rt.pivot.y;
        float top = parent.yMax - panel.height * (1f - rt.pivot.y);

        float x = Mathf.Clamp(desiredAnchoredPos.x, left, right);
        float y = Mathf.Clamp(desiredAnchoredPos.y, bottom, top);
        return new Vector2(x, y);
    }
}
