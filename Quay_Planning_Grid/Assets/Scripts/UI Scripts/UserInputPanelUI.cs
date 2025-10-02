using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

[RequireComponent(typeof(RectTransform))]
public class UserInputPanelUI : MonoBehaviour,
    IPointerDownHandler, IBeginDragHandler, IDragHandler, IEndDragHandler,
    IPointerEnterHandler, IPointerExitHandler
{
    [Header("Refs")]
    public Canvas canvas;
    public CanvasGroup canvasGroup;
    public Shadow shadow;

    [Header("Hover Settings")]
    public float hoverScale = 1.03f;
    public float hoverAlpha = 0.95f;
    public bool bringToFrontOnDrag = true;

    [Header("Bounds")]
    public bool clampToParent = true;

    [Header("Inputs")]
    public TextMeshProUGUI textNumberOfWalls;
    public TextMeshProUGUI textNumberOfShips;
    public Slider NumberOfShips;

    public int SelectedNumberOfShips;

    private RectTransform rt;
    private RectTransform parentRt;
    Vector2 grabOffset;
    private Vector3 originalScale = Vector3.one;
    private float originalAlpha = 1f;

    public bool useOwnSubCanvas = true;
    Canvas selfCanvas;
    static int s_OrderSeed = 100;

    void Awake()
    {
        rt = GetComponent<RectTransform>();
        if (canvas == null) canvas = GetComponentInParent<Canvas>();
        parentRt = rt.parent as RectTransform;

        if (canvasGroup == null)
        {
            canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null) canvasGroup = gameObject.AddComponent<CanvasGroup>();
        }
        originalScale = rt.localScale;
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

    private void Start()
    {
        UpdateNumberOfSelectedShips(NumberOfShips.value);
        NumberOfShips.onValueChanged.AddListener(UpdateNumberOfSelectedShips);
        SelectedNumberOfShips = (int)NumberOfShips.value;
    }

    public void OnPointerEnter(PointerEventData eventData)
    {
        BringToFront();
        rt.localScale = Vector3.Lerp(rt.localScale, originalScale * hoverScale, 1f);
        if (shadow != null) shadow.enabled = true;
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        if (!eventData.dragging)
        {
            rt.localScale = originalScale;
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

    public void UpdateNumberOfSelectedShips(float value)
    {
        if (textNumberOfShips != null)
        {
            // 10 단위로 반올림
            float steppedValue = Mathf.Round(value / 10f) * 10f;

            // 슬라이더 값을 강제로 갱신
            NumberOfShips.SetValueWithoutNotify(steppedValue);
            textNumberOfShips.text = steppedValue.ToString("F0");
        }
    }
}
